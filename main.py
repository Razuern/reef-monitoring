"""Train, evaluate, and use ConvNeXt V2 Base for coral health classification with binary classification methodology.

Expected dataset structure::

    dataset/
    ├── training/
    │   ├── bleached_corals/*.jpg
    │   └── healthy_corals/*.jpg
    ├── validation/
    │   ├── bleached_corals/*.jpg
    │   └── healthy_corals/*.jpg
    └── testing/
        ├── bleached_corals/*.jpg
        └── healthy_corals/*.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


MODEL_NAME = "convnextv2_base"
CLASS_TO_IDX = {"bleached_corals": 0, "healthy_corals": 1}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CoralDataset(Dataset):
    """Dataset whose binary labels are inferred from the class directories based on their names."""

    def __init__(self, split_directory: str | Path, transform: Any = None) -> None:
        self.split_directory = Path(split_directory)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.split_directory.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.split_directory}")

        for class_name, class_index in CLASS_TO_IDX.items():
            class_directory = self.split_directory / class_name
            if not class_directory.is_dir():
                raise FileNotFoundError(f"Expected class directory not found: {class_directory}")

            class_images = sorted(
                path
                for path in class_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not class_images:
                raise RuntimeError(f"No supported images found in: {class_directory}")
            self.samples.extend((path, class_index) for path in class_images)

        if not self.samples:
            raise RuntimeError(f"No images found in: {self.split_directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Could not read image: {image_path}") from error

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    "Return training and deterministic evaluation transforms."

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def create_model(drop_path_rate: float = 0.1) -> nn.Module:
    """Create ConvNeXt V2 Base with random initialization and two outputs."""

    return timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(CLASS_TO_IDX),
        drop_path_rate=drop_path_rate,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    """Compute metrics with bleached coral (class 0) as the positive class."""

    true_positive = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    true_negative = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    false_positive = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    false_negative = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    total = len(labels)

    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision_bleached": precision,
        "recall_bleached": recall,
        "f1_bleached": f1,
        "true_bleached_predicted_bleached": true_positive,
        "true_healthy_predicted_healthy": true_negative,
        "true_healthy_predicted_bleached": false_positive,
        "true_bleached_predicted_healthy": false_negative,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str = "Evaluating",
) -> dict[str, float | int]:
    """Evaluate an already loaded model and return loss and classification metrics."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in tqdm(data_loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = _binary_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / total_examples if total_examples else 0.0
    metrics["number_of_images"] = total_examples
    return metrics


def _class_weights(dataset: CoralDataset, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=len(CLASS_TO_IDX)).float()
    if torch.any(counts == 0):
        raise RuntimeError("Every training class must contain at least one image.")
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    validation_metrics: dict[str, float | int],
    image_size: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "class_to_idx": CLASS_TO_IDX,
            "image_size": image_size,
        },
        checkpoint_path,
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Re-create ConvNeXt V2 Base and load locally trained checkpoint weights."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    checkpoint_mapping = checkpoint.get("class_to_idx")
    if checkpoint_mapping is not None and checkpoint_mapping != CLASS_TO_IDX:
        raise ValueError(
            f"Checkpoint class mapping {checkpoint_mapping} does not match {CLASS_TO_IDX}."
        )
    model = create_model(drop_path_rate=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def train_model(
    dataset_root: str | Path = "dataset",
    output_directory: str | Path = "outputs",
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    image_size: int = 224,
    num_workers: int = 4,
    patience: int = 8,
    seed: int = 42,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Train randomly initialized ConvNeXt V2 Base and save the best checkpoint."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    set_seed(seed)
    device = resolve_device(device_name)
    dataset_root = Path(dataset_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    train_transform, evaluation_transform = build_transforms(image_size)
    training_dataset = CoralDataset(dataset_root / "training", train_transform)
    validation_dataset = CoralDataset(dataset_root / "validation", evaluation_transform)
    training_loader = make_loader(training_dataset, batch_size, True, num_workers, device)
    validation_loader = make_loader(validation_dataset, batch_size, False, num_workers, device)

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(training_dataset, device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint = output_directory / "best_model.pt"
    history: list[dict[str, Any]] = []

    print(f"Device: {device}")
    print(f"Training images: {len(training_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        progress = tqdm(training_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = labels.size(0)
            running_loss += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += batch_count
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        scheduler.step()
        training_loss = running_loss / seen
        training_accuracy = correct / seen
        validation_metrics = evaluate_model(
            model, validation_loader, criterion, device, "Validating"
        )
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                validation_metrics,
                image_size,
            )
            print(f"Saved best checkpoint to {best_checkpoint}")
        #else:
            #epochs_without_improvement += 1
            #if patience > 0 and epochs_without_improvement >= patience:
            #    print(f"Early stopping after {epoch} epochs.")
            #    break

    history_path = output_directory / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "best_checkpoint": str(best_checkpoint),
        "history_file": str(history_path),
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
    }


def evaluate(
    checkpoint_path: str | Path,
    dataset_root: str | Path = "dataset",
    split: str = "testing",
    batch_size: int = 8,
    num_workers: int = 4,
    device_name: str | None = None,
) -> dict[str, float | int]:
    """Load a checkpoint and evaluate it on validation or testing images."""

    if split not in {"validation", "testing"}:
        raise ValueError("split must be either 'validation' or 'testing'.")
    device = resolve_device(device_name)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    image_size = int(checkpoint.get("image_size", 224))
    _, evaluation_transform = build_transforms(image_size)
    dataset = CoralDataset(Path(dataset_root) / split, evaluation_transform)
    loader = make_loader(dataset, batch_size, False, num_workers, device)
    criterion = nn.CrossEntropyLoss()
    return evaluate_model(model, loader, criterion, device, f"Evaluating {split}")


def _find_prediction_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"No supported images found in: {input_path}")
        return images
    raise FileNotFoundError(f"Prediction input not found: {input_path}")


@torch.inference_mode()
def predict(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_csv: str | Path | None = None,
    device_name: str | None = None,
) -> list[dict[str, str | float]]:
    """Predict one image or all images recursively inside a directory."""

    device = resolve_device(device_name)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    image_size = int(checkpoint.get("image_size", 224))
    _, prediction_transform = build_transforms(image_size)
    image_paths = _find_prediction_images(Path(input_path))
    results: list[dict[str, str | float]] = []

    for image_path in tqdm(image_paths, desc="Predicting", leave=False):
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Could not read image: {image_path}") from error

        tensor = prediction_transform(image).unsqueeze(0).to(device)
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
        predicted_index = int(probabilities.argmax().item())
        results.append(
            {
                "image": str(image_path),
                "predicted_class": IDX_TO_CLASS[predicted_index],
                "confidence": float(probabilities[predicted_index].item()),
                "probability_bleached": float(probabilities[CLASS_TO_IDX["bleached_corals"]].item()),
                "probability_healthy": float(probabilities[CLASS_TO_IDX["healthy_corals"]].item()),
            }
        )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train ConvNeXt V2 Base.")
    train_parser.add_argument("--dataset", default="dataset")
    train_parser.add_argument("--output", default="outputs")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default=None, help="Examples: cuda, cuda:0, cpu")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint.")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--dataset", default="dataset")
    evaluate_parser.add_argument("--split", choices=("validation", "testing"), default="testing")
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--num-workers", type=int, default=4)
    evaluate_parser.add_argument("--device", default=None)

    predict_parser = subparsers.add_parser("predict", help="Predict an image or directory.")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output-csv", default='outputs/predictions.csv')
    predict_parser.add_argument("--device", default='cuda')

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.command == "train":
        result = train_model(
            dataset_root=args.dataset,
            output_directory=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            num_workers=args.num_workers,
            patience=args.patience,
            seed=args.seed,
            device_name=args.device,
        )
    elif args.command == "evaluate":
        result = evaluate(
            checkpoint_path=args.checkpoint,
            dataset_root=args.dataset,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
        )
    else:
        result = predict(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            output_csv=args.output_csv,
            device_name=args.device,
        )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
