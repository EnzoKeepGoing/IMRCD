# IMRCD

This repository contains the reproducible implementation of **IMRCD:
Interaction Memory Retrieval for Cognitive Diagnosis**. It includes the
complete model, data preprocessing, five-fold training, validation-based
model selection, and evaluation on a fixed test set.

The public release contains only the complete model and its
reproducibility pipeline. Ablation switches, visualization utilities,
experiment logs, and intermediate files are excluded.

## Repository structure

```text
IMRCD-main/
|-- configs/
|   `-- main_experiments.csv
|-- data/
|   |-- THINK/
|   `-- README.md
|-- imrcd/
|   |-- data/
|   |   `-- dataset.py
|   `-- models/
|       `-- imrcd.py
|-- preprocess_dataset.py
|-- run_experiments.py
|-- train_imrcd.py
|-- pyproject.toml
`-- requirements.txt
```

## Installation

Python 3.8 or later is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Data preprocessing

The preprocessing script retains the first response to repeated
student-exercise interactions, removes students with fewer than 15
interactions, maps all identifiers to zero-based indices, reserves 20% of
each student's interactions as a fixed test set, and assigns the remaining
interactions to five folds.

```bash
python preprocess_dataset.py \
  --input path/to/raw_interactions.csv \
  --output-dir data/THINK \
  --user-column user_id \
  --exercise-column question_id \
  --response-column is_correct \
  --concept-column knowledge_list \
  --timestamp-column created_at \
  --minimum-interactions 15 \
  --test-ratio 0.2 \
  --fold-count 5 \
  --seed 42
```

See `data/README.md` for the processed file format.

## Train one fold

The following command uses the validation-selected configuration for fold 0
of THINK:

```bash
python train_imrcd.py \
  --dataset THINK \
  --data-root data \
  --fold 0 \
  --embedding-dim 80 \
  --interaction-dim 8 \
  --memory-slots 16 \
  --residual-scale 0.01 \
  --query-gamma-max 0.08 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --epochs 30 \
  --patience 5 \
  --num-workers 18 \
  --device cuda
```

The checkpoint with the highest validation AUC is used to compute AUC, ACC,
and RMSE on the fixed test set.

## Reproduce the main experiments

The validation-selected configurations for the five THINK folds are stored
in `configs/main_experiments.csv`.

```bash
python run_experiments.py \
  --config configs/main_experiments.csv \
  --data-root data \
  --output-dir outputs \
  --num-workers 18 \
  --device cuda
```

Each fold produces `best_model.pt` and `metrics.json`. Per-fold results are
written to `outputs/fold_results.csv`, and the mean and standard deviation
over the five folds are written to `outputs/dataset_summary.csv`.

## Saved model weights

For each fold, the training script saves the model with the highest
validation AUC. The default output structure is:

```text
outputs/
|-- THINK/
|   |-- fold_0/
|   |   |-- best_model.pt
|   |   `-- metrics.json
|   |-- fold_1/
|   |   |-- best_model.pt
|   |   `-- metrics.json
|   |-- fold_2/
|   |   |-- best_model.pt
|   |   `-- metrics.json
|   |-- fold_3/
|   |   |-- best_model.pt
|   |   `-- metrics.json
|   `-- fold_4/
|       |-- best_model.pt
|       `-- metrics.json
|-- fold_results.csv
`-- dataset_summary.csv
```

Each `best_model.pt` file contains the model `state_dict`. The corresponding
model and optimization settings, selected epoch, validation AUC, and test
metrics are recorded in `metrics.json`. A checkpoint must be loaded into an
IMRCD model constructed with the settings of the same fold in
`configs/main_experiments.csv`.

Example checkpoint loading:

```python
import torch

from imrcd.models import IMRCD


model = IMRCD(
    student_count=534,
    exercise_count=8027,
    concept_count=844,
    embedding_dim=80,
    interaction_dim=8,
    memory_slots=16,
    residual_scale=0.01,
    query_gamma_max=0.08,
)
state_dict = torch.load(
    "outputs/THINK/fold_0/best_model.pt",
    map_location="cpu",
)
model.load_state_dict(state_dict)
model.eval()
```

## Dataset statistics

| Statistic | THINK | MATH_G4-5 | MATH_G7 |
|:--|--:|--:|--:|
| # Students | 534 | 962 | 701 |
| # Exercises | 8,027 | 20,460 | 17,944 |
| # KCs | 844 | 1,483 | 931 |
| # Responses | 24,866 | 50,936 | 30,418 |
| *Avg. KCs* | 1.63 | 1.69 | 1.29 |
| *Avg. Qs* | 15.53 | 23.26 | 24.84 |

*Avg. KCs* and *Avg. Qs* denote the number of KCs per exercise and exercises
per KC, respectively.

## Main results

The results below are reported as the mean +/- standard deviation over five
folds. For each dataset and metric, the best and second-best results are shown
in **bold** and <u>underlined</u>, respectively. An asterisk denotes a
statistically significant improvement over the best baseline in AUC
(`p < 0.05`).

| Method | THINK AUC | THINK ACC | THINK RMSE | MATH_G4-5 AUC | MATH_G4-5 ACC | MATH_G4-5 RMSE | MATH_G7 AUC | MATH_G7 ACC | MATH_G7 RMSE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DINA | 0.5402 +/- 0.0006 | 0.4918 +/- 0.0180 | 0.5528 +/- 0.0378 | 0.5462 +/- 0.0026 | 0.5142 +/- 0.0022 | 0.5243 +/- 0.0007 | 0.5535 +/- 0.0033 | 0.5479 +/- 0.0023 | 0.5144 +/- 0.0012 |
| IRT | 0.5697 +/- 0.0006 | 0.6856 +/- 0.0054 | 0.4720 +/- 0.0030 | 0.5687 +/- 0.0093 | 0.6446 +/- 0.0129 | 0.4947 +/- 0.0090 | 0.5689 +/- 0.0099 | 0.6365 +/- 0.0135 | 0.5054 +/- 0.0122 |
| MIRT | 0.6372 +/- 0.0044 | 0.6687 +/- 0.0057 | 0.4850 +/- 0.0091 | 0.6317 +/- 0.0073 | 0.6432 +/- 0.0067 | 0.4935 +/- 0.0112 | 0.6324 +/- 0.0085 | 0.6397 +/- 0.0087 | 0.5037 +/- 0.0153 |
| NeuralCDM | 0.6980 +/- 0.0015 | 0.7191 +/- 0.0020 | 0.4370 +/- 0.0010 | 0.6905 +/- 0.0017 | 0.6832 +/- 0.0027 | 0.4540 +/- 0.0060 | 0.7017 +/- 0.0033 | 0.6874 +/- 0.0009 | 0.4606 +/- 0.0065 |
| KaNCD | 0.7238 +/- 0.0028 | 0.7237 +/- 0.0035 | 0.4307 +/- 0.0052 | 0.7064 +/- 0.0025 | 0.6966 +/- 0.0024 | 0.4459 +/- 0.0008 | 0.7200 +/- 0.0068 | 0.6980 +/- 0.0062 | 0.4487 +/- 0.0080 |
| RCD | <u>0.7270 +/- 0.0089</u> | 0.7215 +/- 0.0084 | 0.4355 +/- 0.0094 | 0.6734 +/- 0.0172 | 0.6816 +/- 0.0083 | 0.4713 +/- 0.0289 | 0.7212 +/- 0.0026 | 0.6957 +/- 0.0037 | 0.4565 +/- 0.0022 |
| HyperCDM | 0.6862 +/- 0.0052 | 0.6991 +/- 0.0078 | 0.4657 +/- 0.0039 | 0.6657 +/- 0.0023 | 0.6643 +/- 0.0033 | 0.4758 +/- 0.0187 | 0.6699 +/- 0.0023 | 0.6619 +/- 0.0022 | 0.4647 +/- 0.0038 |
| KAN2CD | 0.7214 +/- 0.0019 | <u>0.7299 +/- 0.0031</u> | <u>0.4280 +/- 0.0013</u> | <u>0.7143 +/- 0.0022</u> | <u>0.7023 +/- 0.0007</u> | <u>0.4422 +/- 0.0010</u> | <u>0.7249 +/- 0.0013</u> | <u>0.7119 +/- 0.0041</u> | <u>0.4405 +/- 0.0006</u> |
| KCD | 0.6970 +/- 0.0014 | 0.7176 +/- 0.0028 | 0.4351 +/- 0.0007 | 0.6888 +/- 0.0028 | 0.6847 +/- 0.0030 | 0.4515 +/- 0.0031 | 0.7023 +/- 0.0032 | 0.6893 +/- 0.0020 | 0.4485 +/- 0.0003 |
| **IMRCD** | **0.7445 +/- 0.0007**<sup>*</sup> | **0.7340 +/- 0.0032** | **0.4196 +/- 0.0013** | **0.7299 +/- 0.0007**<sup>*</sup> | **0.7063 +/- 0.0050** | **0.4376 +/- 0.0011** | **0.7436 +/- 0.0009**<sup>*</sup> | **0.7123 +/- 0.0063** | **0.4309 +/- 0.0080** |

## Performance across exercise accuracy ranges

The following results report AUC as the mean +/- standard deviation over five
folds on the fixed test set. The best result in each setting is shown in
**bold**.

| Dataset | Exercise Accuracy Range | RCD | HyperCDM | KCD | IMRCD |
|:--|:--|--:|--:|--:|--:|
| THINK | Below 30% | 0.6605 +/- 0.0094 | 0.6394 +/- 0.0200 | 0.6388 +/- 0.0155 | **0.6673 +/- 0.0146** |
| THINK | 30%-70% | 0.6805 +/- 0.0119 | 0.6653 +/- 0.0172 | 0.6629 +/- 0.0100 | **0.6957 +/- 0.0062** |
| THINK | Above 70% | 0.6851 +/- 0.0051 | 0.7355 +/- 0.0105 | 0.7097 +/- 0.0059 | **0.7617 +/- 0.0042** |
| MATH_G4-5 | Below 30% | 0.6307 +/- 0.0208 | 0.5825 +/- 0.0463 | 0.6473 +/- 0.0161 | **0.6789 +/- 0.0075** |
| MATH_G4-5 | 30%-70% | 0.6822 +/- 0.0354 | 0.6171 +/- 0.0541 | 0.6940 +/- 0.0076 | **0.7162 +/- 0.0057** |
| MATH_G4-5 | Above 70% | 0.6611 +/- 0.0288 | 0.6337 +/- 0.0535 | 0.6779 +/- 0.0063 | **0.7365 +/- 0.0042** |
| MATH_G7 | Below 30% | 0.6229 +/- 0.0094 | 0.6233 +/- 0.0125 | 0.6227 +/- 0.0204 | **0.6667 +/- 0.0129** |
| MATH_G7 | 30%-70% | 0.6741 +/- 0.0046 | 0.6141 +/- 0.0094 | 0.6660 +/- 0.0137 | **0.6778 +/- 0.0092** |
| MATH_G7 | Above 70% | 0.7086 +/- 0.0048 | 0.7230 +/- 0.0068 | 0.7508 +/- 0.0017 | **0.7742 +/- 0.0040** |

## Appendix figure

Place the appendix figure at `figures/appendix_figure.png`. After the file is
added, it will be displayed below.

<!-- Replace figures/appendix_figure.png with the final appendix figure. -->
![Appendix figure](figures/appendix_figure.png)

## Core implementation

The complete forward path is implemented in `imrcd/models/imrcd.py`:

1. construct a memory query from the target student and current exercise;
2. read the grouped key-value interaction memory;
3. generate KC updates through memory-guided head-wise interaction;
4. fuse the updates with static KC representations; and
5. predict the response probability.

The module registration and initialization order and the training procedure
match the implementation used for the reported experiments.
