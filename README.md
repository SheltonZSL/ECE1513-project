# ECE1513 Project

## Overview
This project studies the adversarial robustness of `bert-base-uncased` on the SST-2 sentiment classification task. The goal is to compare a standard fine-tuned BERT baseline with simple adversarial data augmentation methods and measure the trade-off between clean accuracy and robustness under adversarial attack.

## What We Implemented
- Baseline BERT fine-tuning on SST-2
- Clean validation evaluation
- Adversarial data generation with TextAttack-based augmenters
  - `WordNet`
  - `CharSwap`
  - `Mixed`
- TextFooler-style attack evaluation
- Small-sample ablation support for faster experiments
- Result table and visualization for current experiments

## Repository Structure
- `src/train.py`: train baseline or augmented BERT models
- `src/evaluate.py`: evaluate clean accuracy
- `src/augment.py`: generate augmented training samples
- `src/attack.py`: run adversarial attacks
- `src/plot_small_scale_results.py`: export the current result table and plot under `docs/`
- `report_en.md`: English report draft
- `report_zh.md`: Chinese report draft

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run
Generate augmented data:

```bash
python src/augment.py --strategy charswap --max_samples 500 --output results/aug_charswap_500.json
python src/augment.py --strategy wordnet --max_samples 500 --output results/aug_wordnet_500.json
```

Train a baseline model:

```bash
python src/train.py --mode baseline --output_dir results/baseline_model
```

Train an augmented model:

```bash
python src/train.py --mode augmented --augmented_data results/aug_charswap_500.json --output_dir results/charswap_model
```

Run clean evaluation:

```bash
python src/evaluate.py --model_dir results/baseline_model
```

Run attack evaluation:

```bash
python src/attack.py --model_dir results/baseline_model --num_samples 50 --seed 42 --disable_use_constraint --output results/baseline_attack_light.json
```

Generate the summary plot and CSV under `docs/`:

```bash
python src/plot_small_scale_results.py
```

## Current Experiment Status
We completed a small-scale comparison with a shared setup:
- training samples: `5000`
- epochs: `1`
- device: `CPU`
- attack samples: `50`
- seed: `42`

## Result Table
| Method | Clean Accuracy | Attack Success Rate |
| --- | ---: | ---: |
| Baseline | 0.8796 | 1.00 |
| CharSwap | 0.8853 | 0.98 |
| WordNet | 0.8658 | 0.94 |

## Visualization
![Small-scale experiment comparison](docs/small_scale_results.png)

## Result Interpretation
Under this small-scale setup:
- `CharSwap` gives the best clean accuracy
- `WordNet` gives the best robustness among the three tested settings
- all models remain highly vulnerable to adversarial attack

## Important Note About Attack Results
The current attack results were produced with:
- `--disable_use_constraint`

This disables the `UniversalSentenceEncoder` semantic constraint in the default TextFooler recipe to avoid the heavy TensorFlow dependency chain. Because of that, these results should be described as a lighter `TextFooler-style` attack rather than the full original TextFooler configuration.

## Next Steps
Possible follow-up work:
- run the same setup for the `mixed` augmentation strategy
- increase epochs from `1` to `3`
- run larger training subsets
- repeat attacks with multiple seeds
- add stronger defenses or additional attack methods

