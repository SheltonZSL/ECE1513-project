"""Evaluate clean accuracy of a fine-tuned BERT model on SST-2 validation set."""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation results JSON",
    )
    args = parser.parse_args()

    if args.output is None:
        model_name = os.path.basename(args.model_dir.rstrip("/\\"))
        args.output = f"results/{model_name}_eval_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Load SST-2 validation set
    val_dataset = load_dataset("glue", "sst2", split="validation")
    sentences = val_dataset["sentence"]
    labels = val_dataset["label"]

    # Run inference in batches
    all_preds = []
    total_batches = (len(sentences) + args.batch_size - 1) // args.batch_size
    for batch_idx, i in enumerate(range(0, len(sentences), args.batch_size)):
        batch_sentences = sentences[i : i + args.batch_size]
        inputs = tokenizer(
            batch_sentences,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)

        if (batch_idx + 1) % 5 == 0 or batch_idx + 1 == total_batches:
            print(f"  Batch {batch_idx + 1}/{total_batches} ({len(all_preds)}/{len(sentences)} samples)")

    # Compute metrics
    acc = accuracy_score(labels, all_preds)
    report = classification_report(
        labels, all_preds, target_names=["negative", "positive"], output_dict=True
    )

    print(f"\nClean Accuracy: {acc:.4f}")
    print(classification_report(labels, all_preds, target_names=["negative", "positive"]))

    results = {
        "model_dir": args.model_dir,
        "clean_accuracy": round(acc, 4),
        "classification_report": report,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
