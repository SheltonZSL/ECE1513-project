import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(eval_path: Path, attack_path: Path):
    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with attack_path.open("r", encoding="utf-8") as f:
        attack_data = json.load(f)

    if "clean_accuracy" in eval_data:
        clean_accuracy = float(eval_data["clean_accuracy"])
    elif "eval_accuracy" in eval_data:
        clean_accuracy = float(eval_data["eval_accuracy"])
    else:
        raise KeyError("Expected clean_accuracy or eval_accuracy in evaluation JSON")

    attack_success_rate = float(attack_data["summary"]["attack_success_rate"])
    accuracy_under_attack = clean_accuracy * (1.0 - attack_success_rate)

    return clean_accuracy, attack_success_rate, accuracy_under_attack


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build lightweight experiment table and plot from actual eval/attack outputs"
    )
    parser.add_argument(
        "--baseline-eval",
        default="results/baseline_5000_model_eval_results.json",
        help="Path to baseline clean evaluation JSON",
    )
    parser.add_argument(
        "--baseline-attack",
        default="results/baseline_5000_attack_light.json",
        help="Path to baseline attack JSON",
    )
    parser.add_argument(
        "--charswap-eval",
        default="results/charswap_500_model_eval_results.json",
        help="Path to charswap clean evaluation JSON",
    )
    parser.add_argument(
        "--charswap-attack",
        default="results/charswap_attack_light.json",
        help="Path to charswap attack JSON",
    )
    parser.add_argument(
        "--wordnet-eval",
        default="results/wordnet_500_model_eval_results.json",
        help="Path to wordnet clean evaluation JSON",
    )
    parser.add_argument(
        "--wordnet-attack",
        default="results/wordnet_attack_light.json",
        help="Path to wordnet attack JSON",
    )
    parser.add_argument(
        "--out-csv",
        default="docs/small_scale_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-fig",
        default="docs/small_scale_results.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    io_map = {
        "Baseline": (
            repo_root / args.baseline_eval,
            repo_root / args.baseline_attack,
        ),
        "CharSwap": (
            repo_root / args.charswap_eval,
            repo_root / args.charswap_attack,
        ),
        "WordNet": (
            repo_root / args.wordnet_eval,
            repo_root / args.wordnet_attack,
        ),
    }

    rows = []
    for method, (eval_path, attack_path) in io_map.items():
        require_file(eval_path)
        require_file(attack_path)
        clean_accuracy, attack_success_rate, accuracy_under_attack = load_metrics(
            eval_path, attack_path
        )
        rows.append(
            {
                "method": method,
                "clean_accuracy": clean_accuracy,
                "attack_success_rate": attack_success_rate,
                "accuracy_under_attack": accuracy_under_attack,
            }
        )

    out_csv = repo_root / args.out_csv
    out_fig = repo_root / args.out_fig
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "clean_accuracy",
                "attack_success_rate",
                "accuracy_under_attack",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    methods = [row["method"] for row in rows]
    clean_acc = [row["clean_accuracy"] for row in rows]
    asr = [row["attack_success_rate"] for row in rows]
    acc_under_attack = [row["accuracy_under_attack"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
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
        axes[1].text(i, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(methods, acc_under_attack, color=colors)
    axes[2].set_title("Accuracy Under Attack")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Accuracy")
    for i, value in enumerate(acc_under_attack):
        axes[2].text(i, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Small-Scale Experiment Comparison")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV to {out_csv}")
    print(f"Saved figure to {out_fig}")


if __name__ == "__main__":
    main()
