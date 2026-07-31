from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_concepts(value: object, separator: str) -> list[str]:
    if isinstance(value, str) and value.strip().startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            parsed = [parsed]
        concepts = [str(item).strip() for item in parsed]
    else:
        concepts = [
            item.strip() for item in str(value).split(separator)
        ]
    return [concept for concept in concepts if concept]


def assign_splits(
    frame: pd.DataFrame,
    test_ratio: float,
    fold_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = np.random.default_rng(seed)
    train_valid_parts = []
    test_parts = []

    for _, student_rows in frame.groupby("user_id", sort=True):
        indices = student_rows.index.to_numpy()
        generator.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_ratio)))
        test_indices = indices[:test_count]
        train_valid_indices = indices[test_count:]
        generator.shuffle(train_valid_indices)

        student_test = frame.loc[test_indices].copy()
        student_test["fold"] = -1
        test_parts.append(student_test)

        student_train_valid = frame.loc[train_valid_indices].copy()
        student_train_valid["fold"] = (
            np.arange(len(student_train_valid)) % fold_count
        )
        train_valid_parts.append(student_train_valid)

    train_valid = pd.concat(
        train_valid_parts, ignore_index=True
    ).sample(frac=1.0, random_state=seed)
    test = pd.concat(
        test_parts, ignore_index=True
    ).sort_values(["user_id", "question_id"])
    return train_valid, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare interaction data for IMRCD"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--user-column", default="user_id")
    parser.add_argument("--exercise-column", default="question_id")
    parser.add_argument("--response-column", default="is_correct")
    parser.add_argument("--concept-column", default="knowledge_list")
    parser.add_argument("--timestamp-column", default="created_at")
    parser.add_argument("--concept-separator", default="_")
    parser.add_argument("--minimum-interactions", type=int, default=15)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    raw = pd.read_csv(arguments.input)
    required_columns = {
        arguments.user_column,
        arguments.exercise_column,
        arguments.response_column,
        arguments.concept_column,
    }
    missing = required_columns.difference(raw.columns)
    if missing:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing)}"
        )

    selected_columns = list(required_columns)
    if arguments.timestamp_column in raw.columns:
        selected_columns.append(arguments.timestamp_column)
    frame = raw[selected_columns].dropna(
        subset=list(required_columns)
    ).copy()
    rename_map = {
        arguments.user_column: "original_user_id",
        arguments.exercise_column: "original_question_id",
        arguments.response_column: "correct",
        arguments.concept_column: "concepts",
    }
    if arguments.timestamp_column in frame.columns:
        rename_map[arguments.timestamp_column] = "timestamp"
    frame = frame.rename(columns=rename_map)
    frame["correct"] = frame["correct"].astype(int)
    if not frame["correct"].isin([0, 1]).all():
        raise ValueError("Response values must be binary")

    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(
        ["original_user_id", "original_question_id"],
        keep="first",
    )
    interaction_counts = frame.groupby(
        "original_user_id"
    ).size()
    retained_students = interaction_counts[
        interaction_counts >= arguments.minimum_interactions
    ].index
    frame = frame[
        frame["original_user_id"].isin(retained_students)
    ].copy()
    if frame.empty:
        raise ValueError(
            "No students remain after the minimum-interaction filter"
        )

    user_values = sorted(frame["original_user_id"].unique().tolist())
    question_values = sorted(
        frame["original_question_id"].unique().tolist()
    )
    user_mapping = {
        str(value): index for index, value in enumerate(user_values)
    }
    question_mapping = {
        str(value): index for index, value in enumerate(question_values)
    }
    frame["user_id"] = frame["original_user_id"].map(
        lambda value: user_mapping[str(value)]
    )
    frame["question_id"] = frame["original_question_id"].map(
        lambda value: question_mapping[str(value)]
    )
    frame["concept_list"] = frame["concepts"].map(
        lambda value: parse_concepts(
            value, arguments.concept_separator
        )
    )

    concept_values = sorted(
        {
            concept
            for concepts in frame["concept_list"]
            for concept in concepts
        }
    )
    concept_mapping = {
        concept: index for index, concept in enumerate(concept_values)
    }
    question_concepts = (
        frame.groupby("question_id")["concept_list"]
        .apply(
            lambda rows: sorted(
                {
                    concept_mapping[concept]
                    for concepts in rows
                    for concept in concepts
                }
            )
        )
        .reset_index(name="concept_ids")
    )

    interactions = frame[
        ["user_id", "question_id", "correct"]
    ].copy()
    train_valid, test = assign_splits(
        interactions,
        arguments.test_ratio,
        arguments.fold_count,
        arguments.seed,
    )

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_valid.to_csv(
        output_dir / "train_valid.csv", index=False
    )
    test.to_csv(output_dir / "test.csv", index=False)
    question_concepts.to_csv(
        output_dir / "q_matrix.csv", index=False
    )
    mappings = {
        "students": user_mapping,
        "questions": question_mapping,
        "concepts": concept_mapping,
    }
    with open(
        output_dir / "id_mapping.json", "w", encoding="utf-8"
    ) as mapping_file:
        json.dump(mappings, mapping_file, ensure_ascii=False, indent=2)

    statistics = {
        "students": len(user_mapping),
        "exercises": len(question_mapping),
        "concepts": len(concept_mapping),
        "train_validation_interactions": len(train_valid),
        "test_interactions": len(test),
        "folds": arguments.fold_count,
        "minimum_interactions": arguments.minimum_interactions,
        "test_ratio": arguments.test_ratio,
        "seed": arguments.seed,
    }
    with open(
        output_dir / "statistics.json", "w", encoding="utf-8"
    ) as statistics_file:
        json.dump(statistics, statistics_file, indent=2)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
