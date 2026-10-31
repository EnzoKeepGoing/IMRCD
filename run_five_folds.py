"""Run the released IMRCD configurations on the three datasets."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

from imrcd.data import build_datasets
from imrcd.models import IMRCD
from train_imrcd import resolve_device, run_experiment, set_seed


DATASETS = ("Grade3", "Math50k", "MidMath")
MODEL_DEFAULTS = {
    "interaction_hidden_dim": 64,
    "interaction_heads": 4,
    "num_basis": 3,
    "memory_dim": 32,
    "memory_groups": 4,
}


def configuration(row, options: argparse.Namespace) -> Namespace:
    return Namespace(
        dataset=row.dataset,
        data_root=options.data_root,
        output_dir=options.output_dir,
        fold=int(row.fold),
        seed=int(row.seed),
        device=options.device,
        num_workers=options.num_workers,
        embedding_dim=int(row.embedding_dim),
        interaction_dim=int(row.interaction_dim),
        memory_slots=int(row.memory_slots),
        residual_scale=float(row.residual_scale),
        query_gamma_max=float(row.query_gamma_max),
        learning_rate=float(row.learning_rate),
        batch_size=int(row.batch_size),
        micro_batch_size=int(row.micro_batch_size),
        epochs=30,
        patience=5,
        validation_only=False,
        **MODEL_DEFAULTS,
    )


def dry_run(config: Namespace) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    train, valid, test, info = build_datasets(
        Path(config.data_root) / config.dataset, config.fold
    )
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"Empty split: {config.dataset} fold {config.fold}")
    model = IMRCD(
        student_count=info["student_count"],
        exercise_count=info["exercise_count"],
        concept_count=info["concept_count"],
        embedding_dim=config.embedding_dim,
        interaction_dim=config.interaction_dim,
        memory_slots=config.memory_slots,
        residual_scale=config.residual_scale,
        query_gamma_max=config.query_gamma_max,
        **MODEL_DEFAULTS,
    ).to(device)
    batch = [torch.stack([train[i][field] for i in range(2)]).to(device)
             for field in range(4)]
    prediction = model(batch[0], batch[1], batch[2])
    loss = model.loss(prediction, batch[3])
    loss.backward()
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise ValueError(f"Nonfinite forward/loss: {config.dataset} fold {config.fold}")
    print(f"OK {config.dataset} fold {config.fold}: "
          f"train={len(train)}, valid={len(valid)}, test={len(test)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main_experiments.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_table = pd.read_csv(args.config)
    required = {"dataset", "fold", "seed", "embedding_dim", "interaction_dim",
                "memory_slots", "residual_scale", "query_gamma_max",
                "learning_rate", "batch_size", "micro_batch_size"}
    if required - set(config_table.columns):
        raise ValueError(f"Missing config columns: {sorted(required - set(config_table.columns))}")
    if config_table.duplicated(["dataset", "fold"]).any():
        raise ValueError("Duplicate dataset/fold configuration")
    config_table = config_table[config_table.dataset.isin(args.datasets)]
    for dataset in args.datasets:
        folds = sorted(config_table.loc[config_table.dataset == dataset, "fold"].tolist())
        if folds != list(range(5)):
            raise ValueError(f"Expected folds 0-4 for {dataset}; got {folds}")

    if args.dry_run:
        for row in config_table.itertuples(index=False):
            dry_run(configuration(row, args))
        return

    rows = []
    for row in config_table.itertuples(index=False):
        config = configuration(row, args)
        result_file = Path(args.output_dir) / config.dataset / f"fold_{config.fold}" / "metrics.json"
        if result_file.exists():
            raise FileExistsError(f"Existing result would be overwritten: {result_file}")
        result = run_experiment(config)
        rows.append({"dataset": result["dataset"], "fold": result["fold"],
                     "seed": result["seed"], "test_auc": result["test_auc"],
                     "test_acc": result["test_acc"], "test_rmse": result["test_rmse"]})

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(output / "per_fold_metrics.csv", index=False)
    summary = per_fold.groupby("dataset", as_index=False).agg(
        auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
        acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
        rmse_mean=("test_rmse", "mean"), rmse_std=("test_rmse", "std"),
    )
    summary.to_csv(output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
