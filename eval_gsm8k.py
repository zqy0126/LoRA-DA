import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)


def extract_answer(completion):
    if "####" in completion:
        answer = completion.rsplit("####", 1)[-1].strip()
    elif "The answer is: " in completion:
        answer = completion.rsplit("The answer is: ", 1)[-1].strip()
    else:
        return None

    match = re.search(r"[-+]?\d*[\.,/]?\d+", answer)
    if not match:
        return None
    value = match.group().replace(",", "")
    try:
        return round(float(Fraction(value))) if "/" in value else round(float(value))
    except (ValueError, ZeroDivisionError):
        return None


def load_gsm8k(path):
    prompts = []
    answers = []
    with open(path, encoding="utf-8") as reader:
        for line in reader:
            item = json.loads(line)
            prompts.append(PROMPT.format(instruction=item["question"]))
            answers.append(int(item["answer"].split("#### ")[1].replace(",", "")))
    return prompts, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Merged base model and adapter.")
    parser.add_argument(
        "--data-file",
        default=str(Path(__file__).parent / "data" / "gsm8k_test.jsonl"),
    )
    parser.add_argument("--output", default="gsm8k_results.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    from gsm8k_grader import math_equal
    from vllm import LLM, SamplingParams

    prompts, answers = load_gsm8k(args.data_file)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=args.max_tokens,
        stop=["Instruction:", "Instruction", "Response:", "Response"],
    )
    outputs = llm.generate(prompts, sampling)
    predictions = [extract_answer(output.outputs[0].text) for output in outputs]
    correct = [
        prediction is not None
        and (
            float(prediction) == float(answer)
            or math_equal(prediction, answer)
        )
        for prediction, answer in zip(predictions, answers)
    ]
    result = {
        "model": args.model,
        "prompt": PROMPT,
        "num_examples": len(correct),
        "num_correct": sum(correct),
        "accuracy": sum(correct) / len(correct),
        "num_invalid_outputs": sum(prediction is None for prediction in predictions),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
