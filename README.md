# LoRA-DA

This repository contains the minimal code needed to train a LoRA-DA adapter and
evaluate GSM8K-CoT accuracy.

The default experiment configuration is:

```text
LoRA-One/conf/lora_da.yaml
```

## Training

Install dependencies and run training:

```bash
cd LoRA-One
pip install -r requirements.txt
./train.sh
```

The default training script fine-tunes on MetaMathQA and saves the adapter to:

```text
LoRA-One/safe_results/xxx_meta_math/lora-da-metamath100k-gsm8k-grad256-fisher256-beta0p03-gamma1024-scale1-final/9/final_checkpoint
```

Hydra options can be overridden from the command line. For example:

```bash
./train.sh train_limit=10000 init.stable_gamma=512 init.da_variance_beta=0.0
```

## GSM8K Evaluation

Evaluate a trained adapter with:

```bash
cd LoRA-One
./eval.sh safe_results/xxx_meta_math/lora-da-metamath100k-gsm8k-grad256-fisher256-beta0p03-gamma1024-scale1-final/9/final_checkpoint
```

The evaluation script merges the adapter into the base model and evaluates the
merged model on GSM8K-CoT. By default, outputs are written to:

```text
LoRA-One/merged_model
LoRA-One/gsm8k_results.json
```

Custom output paths can be provided as:

```bash
./eval.sh ADAPTER_PATH OUTPUT_JSON MERGED_MODEL_DIR
```
