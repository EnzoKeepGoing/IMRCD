# IMRCD

This repository contains the IMRCD (Interaction Memory Retrieval for Cognitive Diagnosis) model, processed data, per-fold configurations, and a training/evaluation entry point for Grade3, Math50k, and MidMath.

## Repository layout

```text
configs/main_experiments.csv   Per-fold configurations
data/Grade3/                 Processed responses, Q-matrix, ID mapping
data/Math50k/                Processed responses, Q-matrix, ID mapping
data/MidMath/                Processed responses, Q-matrix, ID mapping
imrcd/models/imrcd.py        IMRCD model
imrcd/data/dataset.py        Data loader
train_imrcd.py              One-fold training and evaluation
run_five_folds.py           Sequential five-fold launcher
results/                    Reference per-fold and summary metrics
```

IMRCD uses student IDs, exercise IDs, and the Q-matrix. The processed data are included; no exercise text, semantic embeddings, or API calls are needed.

## Environment

Install Python, then install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build compatible with your CUDA runtime. The scripts also accept `--device cpu`.

## Data and splits

Each dataset directory contains `train_valid.csv`, `test.csv`, `q_matrix.csv`, and `id_mapping.json`. Response rows use `user_id,question_id,correct,fold`; the Q-matrix maps each `question_id` to its `concept_ids`. Identifiers are zero-based. Within each fold, the matching `fold` value in `train_valid.csv` is the validation set, and the other four folds are the training set. `test.csv` is the fixed evaluation set.

| Dataset | Students | Exercises | KCs | Responses |
| --- | ---: | ---: | ---: | ---: |
| Grade3 | 534 | 8,027 | 844 | 24,866 |
| Math50k | 962 | 20,460 | 1,483 | 50,936 |
| MidMath | 701 | 17,944 | 931 | 30,418 |

## Train and evaluate

Check all 15 configurations with one forward/backward pass each:

```bash
python run_five_folds.py --dry-run --device cpu
```

Run all three datasets, five folds each, sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python run_five_folds.py --device cuda
```

Run only one dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python run_five_folds.py --datasets Grade3 --device cuda
```

To run a single fold directly, pass the corresponding row of [`configs/main_experiments.csv`](configs/main_experiments.csv) to `train_imrcd.py`. For example, Grade3 fold 0:

```bash
python train_imrcd.py --dataset Grade3 --fold 0 --seed 42 \
  --embedding-dim 120 --interaction-dim 16 --memory-slots 8 \
  --residual-scale 0.49194825613979287 --query-gamma-max 0.08 \
  --learning-rate 0.002 --batch-size 64 --micro-batch-size 32 \
  --epochs 30 --patience 5 --num-workers 4 --device cuda \
  --output-dir runs
```

All folds use interaction hidden dimension 64, four interaction heads, three KAN bases, memory dimension 32, and four memory groups. Training uses Adam, binary cross-entropy, a `ReduceLROnPlateau` scheduler (validation AUC, patience 3, factor 0.5), at most 30 epochs, and early stopping after five epochs without validation-AUC improvement. The checkpoint with the highest validation AUC is evaluated on the fixed test set. AUC, ACC, and RMSE are saved in `runs/<dataset>/fold_<n>/metrics.json` with `best_model.pt`. The launcher also writes per-fold and five-fold summary CSV files to `runs/`.

## Reference results

Five-fold mean ± sample standard deviation on the fixed test set:

| Dataset | AUC (%) | ACC (%) | RMSE |
| --- | ---: | ---: | ---: |
| Grade3 | 74.2670 ± 0.0757 | 73.3809 ± 0.3693 | 0.4205 ± 0.0019 |
| Math50k | 72.9807 ± 0.0719 | 70.5827 ± 0.5084 | 0.4378 ± 0.0009 |
| MidMath | 74.3138 ± 0.1919 | 71.5717 ± 0.2725 | 0.4343 ± 0.0010 |

The full-precision results are in [`results/per_fold_metrics.csv`](results/per_fold_metrics.csv) and [`results/summary.csv`](results/summary.csv).
