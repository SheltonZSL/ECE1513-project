"""Fine-tune BERT on SST-2 (baseline or with augmented data)."""

import argparse
import json
import os

import torch
from datasets import load_dataset, Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["sentence"], truncation=True, padding="max_length", max_length=128
    )


def load_augmented_data(path):
    """Load augmented data JSON and return a HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    from datasets import Features, Value, ClassLabel
    features = Features({
        "sentence": Value("string"),
        "label": ClassLabel(names=["negative", "positive"]),
        "idx": Value("int32"),
    })
    return Dataset.from_list(records, features=features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "augmented"],
        default="baseline",
        help="Training mode: baseline (SST-2 only) or augmented (SST-2 + augmented data)",
    )
    parser.add_argument(
        "--augmented_data",
        type=str,
        default="results/augmented_train.json",
        help="Path to augmented training data JSON (used when mode=augmented)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the model (default: results/<mode>_model)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/{args.mode}_model"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    # Load SST-2
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # If augmented mode, merge augmented data into training set
    if args.mode == "augmented":
        print(f"Loading augmented data from {args.augmented_data}")
        aug_dataset = load_augmented_data(args.augmented_data)
        from datasets import concatenate_datasets

        train_dataset = concatenate_datasets([train_dataset, aug_dataset])
        print(f"Combined training set size: {len(train_dataset)}")

    # Tokenize
    print("Tokenizing training set...")
    train_dataset = train_dataset.map(
        lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["sentence", "idx"],
        desc="Tokenizing train",
    )
    print("Tokenizing validation set...")
    val_dataset = val_dataset.map(
        lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["sentence", "idx"],
        desc="Tokenizing val",
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"Starting {args.mode} training...")
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate and save metrics
    metrics = trainer.evaluate()
    print(f"Validation metrics: {metrics}")

    os.makedirs("results", exist_ok=True)
    metrics_path = f"results/{args.mode}_train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
