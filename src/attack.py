"""Run TextFooler attack on a fine-tuned BERT model."""

import argparse
import json
import os

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset as TADataset
from textattack import Attacker, AttackArgs


def get_correctly_classified_samples(model_wrapper, dataset, num_samples=200):
    """Select samples that the model classifies correctly."""
    correct_samples = []
    checked = 0
    for example in dataset:
        sentence = example["sentence"]
        label = example["label"]
        pred = model_wrapper([sentence])[0].argmax()
        checked += 1
        if pred == label:
            correct_samples.append((sentence, label))
        if checked % 50 == 0:
            print(f"  Checked {checked} samples, found {len(correct_samples)}/{num_samples} correct...")
        if len(correct_samples) >= num_samples:
            break
    return correct_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of correctly-classified samples to attack",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for attack results JSON",
    )
    args = parser.parse_args()

    if args.output is None:
        model_name = os.path.basename(args.model_dir.rstrip("/\\"))
        args.output = f"results/{model_name}_attack_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Load validation set and pick correctly-classified samples
    val_dataset = load_dataset("glue", "sst2", split="validation")
    print(f"Selecting {args.num_samples} correctly-classified samples...")

    correct_samples = get_correctly_classified_samples(
        model_wrapper, val_dataset, args.num_samples
    )
    print(f"Found {len(correct_samples)} correctly-classified samples.")

    # Build TextAttack dataset
    ta_dataset = TADataset(
        [(text, label) for text, label in correct_samples],
        input_columns=["text"],
        label_names=["negative", "positive"],
    )

    # Build TextFooler attack
    attack = TextFoolerJin2019.build(model_wrapper)

    # Run attack
    attack_args = AttackArgs(
        num_examples=len(correct_samples),
        log_to_csv=None,
        checkpoint_interval=None,
        disable_stdout=False,
    )
    attacker = Attacker(attack, ta_dataset, attack_args)
    results = attacker.attack_dataset()

    # Compute metrics
    num_successful = sum(
        1 for r in results if isinstance(r, textattack.attack_results.SuccessfulAttackResult)
    )
    num_failed = sum(
        1 for r in results if isinstance(r, textattack.attack_results.FailedAttackResult)
    )
    num_skipped = sum(
        1 for r in results if isinstance(r, textattack.attack_results.SkippedAttackResult)
    )
    total_attacked = num_successful + num_failed
    asr = num_successful / total_attacked if total_attacked > 0 else 0.0

    summary = {
        "model_dir": args.model_dir,
        "num_samples": len(correct_samples),
        "num_successful_attacks": num_successful,
        "num_failed_attacks": num_failed,
        "num_skipped": num_skipped,
        "attack_success_rate": round(asr, 4),
    }

    print("\n=== Attack Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save detailed results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    detailed_results = []
    for r in results:
        entry = {
            "original": r.original_text(),
            "perturbed": r.perturbed_text() if hasattr(r, "perturbed_result") else None,
            "original_output": int(r.original_result.output),
            "result_type": type(r).__name__,
        }
        if isinstance(r, textattack.attack_results.SuccessfulAttackResult):
            entry["perturbed_output"] = int(r.perturbed_result.output)
            entry["num_queries"] = r.num_queries
        detailed_results.append(entry)

    output_data = {"summary": summary, "details": detailed_results}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
