import math
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


def _stable_seed(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def _canonicalize_columns(matrix):
    if matrix.numel() == 0:
        return matrix
    pivot = matrix.abs().argmax(dim=0)
    signs = matrix[pivot, torch.arange(matrix.shape[1], device=matrix.device)].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return matrix * signs.unsqueeze(0)


def _balance_lora_factors(lora_a, lora_b):
    u, singular_values, vh = torch.linalg.svd(lora_b, full_matrices=False)
    pivot = u.abs().argmax(dim=0)
    signs = u[pivot, torch.arange(u.shape[1], device=u.device)].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    u = u * signs.unsqueeze(0)
    vh = vh * signs.unsqueeze(1)
    sqrt_s = singular_values.clamp_min(0).sqrt()
    balanced_b = u * sqrt_s.unsqueeze(0)
    balanced_a = (sqrt_s.unsqueeze(1) * vh) @ lora_a
    return balanced_a, balanced_b


def _deterministic_svd_lowrank(matrix, q, niter, *seed_parts):
    devices = [matrix.device] if matrix.is_cuda else []
    seed = _stable_seed("lora_da_svd_lowrank", matrix.shape, q, niter, *seed_parts)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if matrix.is_cuda:
            torch.cuda.manual_seed_all(seed)
        U, S, V = torch.svd_lowrank(matrix, q=q, niter=niter)
    U = _canonicalize_columns(U)
    V = _canonicalize_columns(V)
    return U, S, V


def _deterministic_lobpcg(matrix, rank, largest, niter, *seed_parts):
    seed = _stable_seed("lora_da_lobpcg", matrix.shape, rank, largest, niter, *seed_parts)
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(seed)
    initial = torch.randn(
        matrix.shape[0],
        rank,
        device=matrix.device,
        dtype=matrix.dtype,
        generator=generator,
    )
    initial, _ = torch.linalg.qr(initial, mode="reduced")
    eigenvalues, eigenvectors = torch.lobpcg(
        matrix,
        k=rank,
        X=initial,
        largest=largest,
        niter=niter,
    )
    eigenvectors = _canonicalize_columns(eigenvectors)
    return eigenvalues, eigenvectors


@dataclass
class _InputFactor:
    value: torch.Tensor
    tokens: int = 0


@dataclass
class _OutputFactor:
    value: torch.Tensor
    tokens: int = 0
    samples: list | None = None


def _input_factor_name(module_name):
    if module_name.endswith(
        (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")
    ):
        return module_name.rsplit(".", 1)[0] + ".qkv_input"
    if module_name.endswith((".mlp.gate_proj", ".mlp.up_proj")):
        return module_name.rsplit(".", 1)[0] + ".gate_up_input"
    return module_name + ".input"


def _layer_number(module_name):
    match = _LAYER_PATTERN.match(module_name)
    return int(match.group(1))


def _select_valid_tokens(tensor, attention_mask):
    return tensor[attention_mask]


def _causal_label_token_mask(labels):
    token_mask = torch.zeros_like(labels, dtype=torch.bool)
    token_mask[:, :-1] = labels[:, 1:].ne(-100)
    return token_mask


class _KFACCollector:
    def __init__(self, named_modules, module_names, device, output_fisher_mode="diagonal"):
        self.device = device
        self.token_mask = None
        self.handles = []
        self.input_factors = {}
        self.output_factors = {}
        self.output_fisher_mode = output_fisher_mode

        for module_name in module_names:
            module = named_modules[module_name]
            factor_name = _input_factor_name(module_name)
            if factor_name not in self.input_factors:
                self.input_factors[factor_name] = _InputFactor(
                    torch.zeros(
                        (module.in_features, module.in_features),
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._make_input_hook(factor_name)
                    )
                )
            self.output_factors[module_name] = _OutputFactor(
                torch.zeros(module.out_features, dtype=torch.float32, device=device),
                samples=[] if output_fisher_mode == "sequence_lowrank" else None,
            )
            self.handles.append(
                module.register_forward_hook(self._make_output_hook(module_name))
            )

    def _make_input_hook(self, factor_name):
        def input_hook(_module, inputs):
            activation = _select_valid_tokens(
                inputs[0].detach(), self.token_mask
            ).float()
            factor = self.input_factors[factor_name]
            factor.value.addmm_(activation.T, activation)
            factor.tokens += activation.shape[0]

        return input_hook

    def _make_output_hook(self, module_name):
        def output_hook(_module, _inputs, output):
            def gradient_hook(gradient):
                gradient = _select_valid_tokens(
                    gradient.detach(), self.token_mask
                ).float()
                factor = self.output_factors[module_name]
                factor.value.add_(gradient.square().sum(dim=0))
                factor.tokens += gradient.shape[0]
                if factor.samples is not None:
                    factor.samples.append(gradient.sum(dim=0))

            output.register_hook(gradient_hook)

        return output_hook

    def set_token_mask(self, token_mask):
        self.token_mask = token_mask.to(device=self.device, dtype=torch.bool)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.token_mask = None


def _make_layer_passes(module_names, layers_per_pass):
    modules_by_layer = defaultdict(list)
    for module_name in module_names:
        modules_by_layer[_layer_number(module_name)].append(module_name)

    layers = sorted(modules_by_layer)
    return [
        [
            module_name
            for layer in layers[start : start + layers_per_pass]
            for module_name in modules_by_layer[layer]
        ]
        for start in range(0, len(layers), layers_per_pass)
    ]


def _require_input_grad(_module, _inputs, output):
    output.requires_grad_(True)


def _compute_group_initializations_for_betas(
    collector,
    grouped_modules,
    named_grads,
    rank,
    bias_alpha,
    variance_betas,
    population_size,
    damping,
    gamma,
    lobpcg_niter,
    normalize_delta,
    delta_svd_niter,
    delta_normalization_power=1.0,
    delta_normalization_reference=1.0,
    deterministic_seed=0,
    balance_factors=False,
    output_damping_ratio=None,
    normalize_variance_guidance=False,
    output_fisher_mode="diagonal",
):
    initializations = {variance_beta: {} for variance_beta in variance_betas}
    for factor_name, module_names in grouped_modules.items():
        input_factor = collector.input_factors[factor_name]
        z_fisher = input_factor.value / input_factor.tokens
        z_fisher = (z_fisher + z_fisher.T) / 2
        z_fisher.diagonal().add_(damping)
        chol, info = torch.linalg.cholesky_ex(z_fisher)

        z_inverse = None
        if any(variance_beta != 0 for variance_beta in variance_betas):
            z_inverse = torch.cholesky_inverse(chol)

        for module_name in module_names:
            output_factor = collector.output_factors[module_name]
            y_fisher_diag = output_factor.value / output_factor.tokens
            y_damping = damping
            if output_damping_ratio is not None:
                y_damping = max(
                    output_damping_ratio * y_fisher_diag.median().item(),
                    torch.finfo(y_fisher_diag.dtype).tiny,
                )
            if output_fisher_mode == "sequence_lowrank":
                sample_gradients = torch.stack(output_factor.samples)
                sample_gradients = sample_gradients / math.sqrt(
                    sample_gradients.shape[0]
                )
                gram = sample_gradients @ sample_gradients.T
                gram.diagonal().add_(y_damping)
                gram_chol, info = torch.linalg.cholesky_ex(gram)
                solved = torch.cholesky_solve(sample_gradients, gram_chol)
                correction = (sample_gradients * solved).sum(dim=0)
                y_inverse_diag = (1 - correction).clamp_min(0) / y_damping
                del sample_gradients, gram, gram_chol, solved, correction
            else:
                y_inverse_diag = (y_fisher_diag + y_damping).reciprocal()

            grad_name = module_name + ".weight"
            gradient = named_grads[grad_name].to(
                device=chol.device, dtype=torch.float32
            )
            delta_w = -torch.cholesky_solve(gradient.T, chol)
            delta_w.mul_(y_inverse_diag.unsqueeze(0))

            guidance_scale = None
            if normalize_delta:
                _, singular_values, _ = _deterministic_svd_lowrank(
                    delta_w,
                    1,
                    delta_svd_niter,
                    deterministic_seed,
                    module_name,
                    "delta",
                )
                sigma_max = singular_values[0]
                normalization_scale = sigma_max.pow(delta_normalization_power)
                normalization_scale *= delta_normalization_reference ** (
                    1.0 - delta_normalization_power
                )
                delta_w.div_(normalization_scale)
                guidance_scale = normalization_scale
            bias_guidance = -bias_alpha * (delta_w @ delta_w.T)
            variance_coefficient = y_inverse_diag.sum().item() / population_size
            if guidance_scale is not None:
                variance_coefficient /= guidance_scale.item()
                variance_coefficient /= guidance_scale.item()
            bias_norm = torch.linalg.matrix_norm(bias_guidance).item()
            variance_norm = (
                abs(variance_coefficient) * torch.linalg.matrix_norm(z_inverse).item()
                if z_inverse is not None
                else 0.0
            )
            if normalize_variance_guidance:
                if variance_norm > 0:
                    variance_coefficient *= bias_norm / variance_norm
            for variance_beta in variance_betas:
                guidance = bias_guidance.clone()
                if variance_beta != 0:
                    guidance.add_(
                        z_inverse,
                        alpha=variance_beta * variance_coefficient,
                    )
                guidance = (guidance + guidance.T) / 2

                _, a0 = _deterministic_lobpcg(
                    guidance,
                    rank,
                    False,
                    lobpcg_niter,
                    deterministic_seed,
                    module_name,
                    variance_beta,
                )
                b0 = a0.T @ delta_w
                lora_a = a0.T
                lora_b = b0.T
                if balance_factors:
                    lora_a, lora_b = _balance_lora_factors(lora_a, lora_b)
                scale = math.sqrt(gamma)
                initializations[variance_beta][grad_name] = {
                    "lora_A": (lora_a / scale).cpu(),
                    "lora_B": (lora_b / scale).cpu(),
                }
                del guidance, a0, b0, lora_a, lora_b

            del gradient, delta_w, bias_guidance

        del z_fisher, chol, z_inverse
        torch.cuda.empty_cache()
    return initializations


def _compute_group_initializations(
    collector,
    grouped_modules,
    named_grads,
    rank,
    bias_alpha,
    variance_beta,
    population_size,
    damping,
    gamma,
    lobpcg_niter,
    normalize_delta,
    delta_svd_niter,
    delta_normalization_power=1.0,
    delta_normalization_reference=1.0,
    deterministic_seed=0,
    balance_factors=False,
    output_damping_ratio=None,
    normalize_variance_guidance=False,
    output_fisher_mode="diagonal",
):
    return _compute_group_initializations_for_betas(
        collector=collector,
        grouped_modules=grouped_modules,
        named_grads=named_grads,
        rank=rank,
        bias_alpha=bias_alpha,
        variance_betas=[variance_beta],
        population_size=population_size,
        damping=damping,
        gamma=gamma,
        lobpcg_niter=lobpcg_niter,
        normalize_delta=normalize_delta,
        delta_svd_niter=delta_svd_niter,
        delta_normalization_power=delta_normalization_power,
        delta_normalization_reference=delta_normalization_reference,
        deterministic_seed=deterministic_seed,
        balance_factors=balance_factors,
        output_damping_ratio=output_damping_ratio,
        normalize_variance_guidance=normalize_variance_guidance,
        output_fisher_mode=output_fisher_mode,
    )[variance_beta]


def estimate_lora_da_initializations_for_betas(
    model,
    dataset,
    named_grads,
    target_modules,
    rank,
    population_size,
    variance_betas,
    batch_size=1,
    bias_alpha=1.0,
    damping=1e-6,
    gamma=32,
    lobpcg_niter=10,
    fisher_layers_per_pass=1,
    normalize_delta=True,
    delta_svd_niter=16,
    delta_normalization_power=1.0,
    delta_normalization_reference=1.0,
    fisher_token_mask="attention",
    deterministic_seed=0,
    balance_factors=False,
    output_damping_ratio=None,
    normalize_variance_guidance=False,
    output_fisher_mode="diagonal",
):
    variance_betas = list(dict.fromkeys(variance_betas))
    named_modules = dict(model.named_modules())
    module_names = sorted(
        name
        for name, module in named_modules.items()
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in target_modules
    )

    layer_passes = _make_layer_passes(module_names, fisher_layers_per_pass)
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    original_training = model.training
    initializations = {variance_beta: {} for variance_beta in variance_betas}

    for parameter in model.parameters():
        parameter.requires_grad = False
    embedding_handle = model.get_input_embeddings().register_forward_hook(
        _require_input_grad
    )
    model.train()

    for pass_index, pass_modules in enumerate(layer_passes, start=1):
        collector = _KFACCollector(
            named_modules,
            pass_modules,
            model.device,
            output_fisher_mode=output_fisher_mode,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        progress = tqdm(
            dataloader,
            desc=f"Estimating Fisher {pass_index}/{len(layer_passes)}",
        )
        for batch in progress:
            model.zero_grad(set_to_none=True)
            batch = {name: value.to(model.device) for name, value in batch.items()}
            token_mask = batch["attention_mask"]
            if fisher_token_mask == "labels":
                token_mask = _causal_label_token_mask(batch["labels"])
            collector.set_token_mask(token_mask)
            model(**batch).loss.backward()

        grouped_modules = defaultdict(list)
        for module_name in pass_modules:
            grouped_modules[_input_factor_name(module_name)].append(module_name)
        stage_initializations = _compute_group_initializations_for_betas(
            collector=collector,
            grouped_modules=grouped_modules,
            named_grads=named_grads,
            rank=rank,
            bias_alpha=bias_alpha,
            variance_betas=variance_betas,
            population_size=population_size,
            damping=damping,
            gamma=gamma,
            lobpcg_niter=lobpcg_niter,
            normalize_delta=normalize_delta,
            delta_svd_niter=delta_svd_niter,
            delta_normalization_power=delta_normalization_power,
            delta_normalization_reference=delta_normalization_reference,
            deterministic_seed=deterministic_seed,
            balance_factors=balance_factors,
            output_damping_ratio=output_damping_ratio,
            normalize_variance_guidance=normalize_variance_guidance,
            output_fisher_mode=output_fisher_mode,
        )
        for variance_beta in variance_betas:
            initializations[variance_beta].update(
                stage_initializations[variance_beta]
            )
        collector.close()
        model.zero_grad(set_to_none=True)
        del collector
        torch.cuda.empty_cache()

    embedding_handle.remove()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = original_requires_grad[name]
    model.train(original_training)
    torch.cuda.empty_cache()

    return initializations


def estimate_lora_da_initializations(
    model,
    dataset,
    named_grads,
    target_modules,
    rank,
    population_size,
    batch_size=1,
    bias_alpha=1.0,
    variance_beta=0.0,
    damping=1e-6,
    gamma=32,
    lobpcg_niter=10,
    fisher_layers_per_pass=1,
    normalize_delta=True,
    delta_svd_niter=16,
    delta_normalization_power=1.0,
    delta_normalization_reference=1.0,
    fisher_token_mask="attention",
    deterministic_seed=0,
    balance_factors=False,
    output_damping_ratio=None,
    normalize_variance_guidance=False,
    output_fisher_mode="diagonal",
):
    return estimate_lora_da_initializations_for_betas(
        model=model,
        dataset=dataset,
        named_grads=named_grads,
        target_modules=target_modules,
        rank=rank,
        population_size=population_size,
        variance_betas=[variance_beta],
        batch_size=batch_size,
        bias_alpha=bias_alpha,
        damping=damping,
        gamma=gamma,
        lobpcg_niter=lobpcg_niter,
        fisher_layers_per_pass=fisher_layers_per_pass,
        normalize_delta=normalize_delta,
        delta_svd_niter=delta_svd_niter,
        delta_normalization_power=delta_normalization_power,
        delta_normalization_reference=delta_normalization_reference,
        fisher_token_mask=fisher_token_mask,
        deterministic_seed=deterministic_seed,
        balance_factors=balance_factors,
        output_damping_ratio=output_damping_ratio,
        normalize_variance_guidance=normalize_variance_guidance,
        output_fisher_mode=output_fisher_mode,
    )[variance_beta]
