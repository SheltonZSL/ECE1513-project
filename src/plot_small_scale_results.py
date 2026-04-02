import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = [
    {"method": "Baseline", "clean_accuracy": 0.8796, "attack_success_rate": 1.00},
    {"method": "CharSwap", "clean_accuracy": 0.8853, "attack_success_rate": 0.98},
    {"method": "WordNet", "clean_accuracy": 0.8658, "attack_success_rate": 0.94},
]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = docs_dir / "small_scale_results.csv"
    fig_path = docs_dir / "small_scale_results.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "clean_accuracy", "attack_success_rate"])
        writer.writeheader()
        writer.writerows(RESULTS)

    methods = [row["method"] for row in RESULTS]
    clean_acc = [row["clean_accuracy"] for row in RESULTS]
    asr = [row["attack_success_rate"] for row in RESULTS]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    colors = ["#355C7D", "#2A9D8F", "#E76F51"]

    axes[0].bar(methods, clean_acc, color=colors)
    axes[0].set_title("Clean Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Accuracy")
    for i, value in enumerate(clean_acc):
        axes[0].text(i, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(methods, asr, color=colors)
    axes[1].set_title("Attack Success Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("ASR")
    for i, value in enumerate(asr):
        axes[1].text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Small-Scale Experiment Comparison")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
