from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_INTERACTION_COLUMNS = {
    "user_id",
    "question_id",
    "correct",
}


def _parse_concept_ids(value: object) -> list[int]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        parsed = [value]

    if not isinstance(parsed, (list, tuple, np.ndarray)):
        parsed = [parsed]
    return [int(concept_id) for concept_id in parsed]


def load_q_matrix(q_matrix_path: str | Path) -> np.ndarray:
    frame = pd.read_csv(q_matrix_path)
    required = {"question_id", "concept_ids"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Q-matrix file is missing columns: {sorted(missing)}"
        )

    exercise_ids = frame["question_id"].astype(int)
    expected_ids = np.arange(len(frame))
    if not np.array_equal(np.sort(exercise_ids.unique()), expected_ids):
        raise ValueError(
            "question_id values in q_matrix.csv must be contiguous "
            "and start at zero"
        )

    parsed_rows = [
        _parse_concept_ids(value) for value in frame["concept_ids"]
    ]
    if any(len(row) == 0 for row in parsed_rows):
        raise ValueError("Every exercise must be associated with at least one KC")

    concept_count = max(
        concept_id for row in parsed_rows for concept_id in row
    ) + 1
    q_matrix = np.zeros(
        (len(frame), concept_count), dtype=np.float32
    )
    for exercise_id, concept_ids in zip(exercise_ids, parsed_rows):
        for concept_id in concept_ids:
            if concept_id < 0:
                raise ValueError("KC identifiers must be nonnegative")
            q_matrix[exercise_id, concept_id] = 1.0
    return q_matrix


def load_interactions(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_INTERACTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Interaction file is missing columns: {sorted(missing)}"
        )
    frame = frame.dropna(
        subset=["user_id", "question_id", "correct"]
    ).copy()
    frame["user_id"] = frame["user_id"].astype(int)
    frame["question_id"] = frame["question_id"].astype(int)
    frame["correct"] = frame["correct"].astype(np.float32)
    if not frame["correct"].isin([0.0, 1.0]).all():
        raise ValueError("correct must contain only binary values")
    return frame


class InteractionDataset(Dataset):
    """Student-exercise interactions with their associated Q-vectors."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        q_matrix: np.ndarray,
    ):
        self.student_ids = interactions["user_id"].to_numpy(
            dtype=np.int64
        )
        self.exercise_ids = interactions["question_id"].to_numpy(
            dtype=np.int64
        )
        self.labels = interactions["correct"].to_numpy(
            dtype=np.float32
        )
        self.q_matrix = torch.from_numpy(q_matrix)

        if len(self.exercise_ids) > 0:
            if self.exercise_ids.min() < 0:
                raise ValueError("Exercise identifiers must be nonnegative")
            if self.exercise_ids.max() >= len(q_matrix):
                raise ValueError(
                    "An interaction references an exercise that is "
                    "absent from q_matrix.csv"
                )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        student_id = self.student_ids[index]
        exercise_id = self.exercise_ids[index]
        return (
            torch.tensor(student_id, dtype=torch.long),
            torch.tensor(exercise_id, dtype=torch.long),
            self.q_matrix[exercise_id],
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


def build_datasets(
    dataset_dir: str | Path,
    fold: int,
) -> tuple[
    InteractionDataset,
    InteractionDataset,
    InteractionDataset,
    dict[str, int],
]:
    dataset_dir = Path(dataset_dir)
    q_matrix = load_q_matrix(dataset_dir / "q_matrix.csv")
    train_valid = load_interactions(dataset_dir / "train_valid.csv")
    test = load_interactions(dataset_dir / "test.csv")

    if "fold" not in train_valid.columns:
        raise ValueError("train_valid.csv must contain a fold column")
    train_valid["fold"] = train_valid["fold"].astype(int)
    available_folds = sorted(train_valid["fold"].unique().tolist())
    if fold not in available_folds:
        raise ValueError(
            f"Fold {fold} is unavailable. Available folds: {available_folds}"
        )

    train_frame = train_valid[train_valid["fold"] != fold].copy()
    valid_frame = train_valid[train_valid["fold"] == fold].copy()
    if train_frame.empty or valid_frame.empty or test.empty:
        raise ValueError("Train, validation, and test sets must be nonempty")

    all_interactions = pd.concat(
        [train_valid, test], ignore_index=True
    )
    student_count = int(all_interactions["user_id"].max()) + 1
    metadata = {
        "student_count": student_count,
        "exercise_count": int(q_matrix.shape[0]),
        "concept_count": int(q_matrix.shape[1]),
        "train_count": int(len(train_frame)),
        "valid_count": int(len(valid_frame)),
        "test_count": int(len(test)),
    }
    return (
        InteractionDataset(train_frame, q_matrix),
        InteractionDataset(valid_frame, q_matrix),
        InteractionDataset(test, q_matrix),
        metadata,
    )
