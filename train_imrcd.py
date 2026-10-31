from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

from imrcd.data import build_datasets
from imrcd.models import IMRCD


def set_seed(seed: int) -> None:
    """Set the random state exactly as in the reported experiments."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    loader_arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_arguments["persistent_workers"] = True
        loader_arguments["prefetch_factor"] = 4
    return DataLoader(**loader_arguments)


def collect_predictions(
    model: IMRCD,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for student_ids, exercise_ids, q_vectors, batch_labels in data_loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            q_vectors = q_vectors.to(device)
            outputs = model(student_ids, exercise_ids, q_vectors)
            predictions.extend(outputs.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())
    return (
        np.asarray(labels, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
    )


def evaluate_auc(
    model: IMRCD,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    labels, predictions = collect_predictions(model, data_loader, device)
    return float(roc_auc_score(labels, predictions))


def evaluate_acc(
    model: IMRCD,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    labels, predictions = collect_predictions(model, data_loader, device)
    predicted_labels = predictions >= 0.5
    return float(accuracy_score(labels, predicted_labels))


def evaluate_rmse(
    model: IMRCD,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    labels, predictions = collect_predictions(model, data_loader, device)
    return float(mean_squared_error(labels, predictions) ** 0.5)


def backward_batch(model, tensors, micro_batch_size=0):
    """Accumulate mean BCE over an effective batch; optimizer steps once outside."""
    size = len(tensors[-1])
    chunk_size = micro_batch_size or size
    batch_loss = 0.0
    for start in range(0, size, chunk_size):
        student, exercise, q, labels = [t[start:start + chunk_size] for t in tensors]
        prediction = model(student, exercise, q)
        loss = model.loss(prediction, labels) * (len(labels) / size)
        loss.backward()
        batch_loss += float(loss.detach().item())
    return batch_loss


def run_experiment(arguments: argparse.Namespace, epoch_callback=None) -> dict:
    set_seed(arguments.seed)
    device = resolve_device(arguments.device)
    dataset_dir = Path(arguments.data_root) / arguments.dataset
    train_dataset, valid_dataset, test_dataset, metadata = build_datasets(
        dataset_dir, arguments.fold
    )

    train_loader = make_loader(
        train_dataset,
        arguments.batch_size,
        True,
        arguments.num_workers,
    )
    micro_batch_size = getattr(arguments, "micro_batch_size", 0)
    evaluation_batch_size = min(arguments.batch_size, micro_batch_size or arguments.batch_size)
    valid_loader = make_loader(
        valid_dataset,
        evaluation_batch_size,
        False,
        arguments.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        evaluation_batch_size,
        False,
        arguments.num_workers,
    )

    model = IMRCD(
        student_count=metadata["student_count"],
        exercise_count=metadata["exercise_count"],
        concept_count=metadata["concept_count"],
        embedding_dim=arguments.embedding_dim,
        interaction_dim=arguments.interaction_dim,
        interaction_hidden_dim=arguments.interaction_hidden_dim,
        interaction_heads=arguments.interaction_heads,
        num_basis=arguments.num_basis,
        memory_dim=arguments.memory_dim,
        memory_groups=arguments.memory_groups,
        memory_slots=arguments.memory_slots,
        residual_scale=arguments.residual_scale,
        query_gamma_max=arguments.query_gamma_max,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=arguments.learning_rate
    )
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5,
    )

    output_dir = (
        Path(arguments.output_dir)
        / arguments.dataset
        / f"fold_{arguments.fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    best_validation_auc: float | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for student_ids, exercise_ids, q_vectors, labels in train_loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            q_vectors = q_vectors.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            batch_loss = backward_batch(model,
                (student_ids, exercise_ids, q_vectors, labels), micro_batch_size)
            optimizer.step()
            batch_losses.append(batch_loss)

        train_loss = sum(batch_losses) / len(batch_losses)

        # The original evaluation obtains each metric in a separate pass.
        validation_auc = evaluate_auc(model, valid_loader, device)
        validation_acc = evaluate_acc(model, valid_loader, device)
        validation_rmse = evaluate_rmse(model, valid_loader, device)
        scheduler.step(validation_auc)
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_auc": validation_auc,
            "validation_acc": validation_acc,
            "validation_rmse": validation_rmse,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.6f} | "
            f"val_auc={validation_auc:.6f} | "
            f"val_acc={validation_acc:.6f} | "
            f"val_rmse={validation_rmse:.6f}"
        )

        improved = (
            best_validation_auc is None
            or validation_auc > best_validation_auc
        )
        if improved:
            best_validation_auc = validation_auc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch_callback is not None:
            epoch_callback(dict(training_history[-1]), best_validation_auc)

        if epochs_without_improvement >= arguments.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if getattr(arguments, "validation_only", False):
        test_auc = test_acc = test_rmse = None
    else:
        test_auc = evaluate_auc(model, test_loader, device)
        test_acc = evaluate_acc(model, test_loader, device)
        test_rmse = evaluate_rmse(model, test_loader, device)

    result = {
        "dataset": arguments.dataset,
        "fold": arguments.fold,
        "seed": arguments.seed,
        "best_epoch": best_epoch,
        "best_validation_auc": best_validation_auc,
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_rmse": test_rmse,
        "checkpoint": str(checkpoint_path),
        "training_history": training_history,
        "data": metadata,
        "model": {
            "embedding_dim": arguments.embedding_dim,
            "interaction_dim": arguments.interaction_dim,
            "interaction_hidden_dim": arguments.interaction_hidden_dim,
            "interaction_heads": arguments.interaction_heads,
            "num_basis": arguments.num_basis,
            "memory_dim": arguments.memory_dim,
            "memory_groups": arguments.memory_groups,
            "memory_slots": arguments.memory_slots,
            "residual_scale": arguments.residual_scale,
            "query_gamma_max": arguments.query_gamma_max,
        },
        "optimization": {
            "learning_rate": arguments.learning_rate,
            "batch_size": arguments.batch_size,
            "micro_batch_size": micro_batch_size,
            "evaluation_batch_size": evaluation_batch_size,
            "epochs": arguments.epochs,
            "patience": arguments.patience,
            "num_workers": arguments.num_workers,
            "scheduler": (
                "ReduceLROnPlateau(mode=max, patience=3, factor=0.5, "
                "stepped with validation AUC)"
            ),
            "test_model": "best validation AUC model",
        },
    }
    with open(
        output_dir / "metrics.json", "w", encoding="utf-8"
    ) as output_file:
        json.dump(result, output_file, indent=2)

    if test_auc is not None:
        print(
            f"Test | AUC={test_auc:.6f} | "
            f"ACC={test_acc:.6f} | "
            f"RMSE={test_rmse:.6f}"
        )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate IMRCD"
    )
    parser.add_argument("--dataset", default="THINK")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip test evaluation while selecting hyperparameters")

    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--interaction-dim", type=int, default=11)
    parser.add_argument("--interaction-hidden-dim", type=int, default=64)
    parser.add_argument("--interaction-heads", type=int, default=4)
    parser.add_argument("--num-basis", type=int, default=3)
    parser.add_argument("--memory-dim", type=int, default=32)
    parser.add_argument("--memory-groups", type=int, default=4)
    parser.add_argument("--memory-slots", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--query-gamma-max", type=float, default=0.05)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    return parser


if __name__ == "__main__":
    run_experiment(build_argument_parser().parse_args())
