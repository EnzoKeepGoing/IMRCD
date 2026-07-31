# Data format

Each processed dataset must be stored in its own directory:

```text
data/
  DATASET_NAME/
    train_valid.csv
    test.csv
    q_matrix.csv
    id_mapping.json
```

`train_valid.csv` contains:

```text
user_id,question_id,correct,fold
```

`test.csv` contains:

```text
user_id,question_id,correct,fold
```

The test fold is represented by `-1`. Student and exercise identifiers
must be zero-based integers. `correct` must be either `0` or `1`.

`q_matrix.csv` contains:

```text
question_id,concept_ids
```

`concept_ids` is a Python-style list of zero-based KC identifiers, for
example, `"[3, 7]"`.

Use `preprocess_dataset.py` to convert a raw interaction CSV into this
format.
