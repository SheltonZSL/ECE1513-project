"""Generate adversarial augmented training data using TextAttack augmenters."""

import argparse
import json
import os
import time

from datasets import load_dataset
from textattack.augmentation import WordNetAugmenter, CharSwapAugmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["wordnet", "charswap", "mixed"],
        default="mixed",
        help="Augmentation strategy to use",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only augment the first N training samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/augmented_train.json",
        help="Output path for augmented data JSON",
    )
    parser.add_argument(
        "--pct_word", type=float, default=0.2,
        help="Fraction of words to substitute via WordNet (default: 0.2)",
    )
    parser.add_argument(
        "--pct_char", type=float, default=0.1,
        help="Fraction of words to perturb via CharSwap (default: 0.1)",
    )
    args = parser.parse_args()

    dataset = load_dataset("glue", "sst2", split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Original training set size: {len(dataset)}")

    wordnet_aug = WordNetAugmenter(
        pct_words_to_swap=args.pct_word,
        transformations_per_example=1,
    )
    charswap_aug = CharSwapAugmenter(
        pct_words_to_swap=args.pct_char,
        transformations_per_example=1,
    )

    augmented_records = []
    total = len(dataset)
    start_time = time.time()

    for i, example in enumerate(dataset):
        sentence = example["sentence"]
        label = example["label"]

        if args.strategy == "wordnet":
            augmenter = wordnet_aug
        elif args.strategy == "charswap":
            augmenter = charswap_aug
        else:
            augmenter = wordnet_aug if i % 2 == 0 else charswap_aug

        try:
            augmented_texts = augmenter.augment(sentence)
            aug_text = augmented_texts[0] if augmented_texts else sentence
        except Exception:
            aug_text = sentence

        augmented_records.append({
            "sentence": aug_text,
            "label": label,
            "idx": total + i,
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(
                f"[{i + 1:>6}/{total}] "
                f"{(i + 1) / total * 100:5.1f}% | "
                f"{rate:.1f} samples/s | "
                f"ETA: {eta / 60:.1f} min"
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(augmented_records, f, indent=2)

    print(f"Done. {len(augmented_records)} augmented examples saved to {args.output}")


if __name__ == "__main__":
    main()
