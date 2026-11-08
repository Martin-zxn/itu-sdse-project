"""Train the vial image classifier with ``python -m modeling.train``."""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from modeling.model import SimpleCNN

IMAGE_SIZE = (64, 64)
TRANSFORM_MEAN = (0.485, 0.456, 0.406)
TRANSFORM_STD = (0.229, 0.224, 0.225)
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_THRESHOLD = 0.5
DEFAULT_RANDOM_SEED = 42

ImageRecord = Tuple[Path, int]


class GlassVialDataset(Dataset):
    """Open vial images and apply the requested preprocessing transform."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform=None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        with Image.open(self.image_paths[index]) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def define_transforms():
    """Use the same augmentation and normalization as the training notebook."""
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
        ]
    )
    return train_transform, validation_transform


def load_image_dataset(
    metadata_path: Path,
    image_data_dir: Path,
) -> List[ImageRecord]:
    """Read ``filename`` and binary ``class`` labels from the metadata CSV."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    records: List[ImageRecord] = []
    missing_files = []
    with metadata_path.open(newline="", encoding="utf-8-sig") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if not {"filename", "class"}.issubset(reader.fieldnames or []):
            raise ValueError("Metadata CSV must contain 'filename' and 'class' columns.")
        for row in reader:
            try:
                label = int(row["class"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid class label for {row.get('filename')!r}."
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f"Class label for {row.get('filename')!r} must be 0 or 1."
                )

            image_path = image_data_dir / row["filename"]
            if image_path.is_file():
                records.append((image_path, label))
            else:
                missing_files.append(row["filename"])

    if not records:
        raise FileNotFoundError(
            f"No images from {metadata_path} were found in {image_data_dir}."
        )
    if missing_files:
        print(f"Skipped {len(missing_files)} metadata rows with missing images.")
    return records


def split_dataset(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """Reproduce the notebook's seeded, per-class 70/15/15 split."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    class_records = {
        label: [record for record in records if record[1] == label]
        for label in (0, 1)
    }

    splits = [[], [], []]
    for label in (0, 1):
        class_items = class_records[label]
        np.random.shuffle(class_items)
        train_size = int(len(class_items) * train_ratio)
        val_size = int(len(class_items) * val_ratio)
        splits[0].extend(class_items[:train_size])
        splits[1].extend(class_items[train_size : train_size + val_size])
        splits[2].extend(class_items[train_size + val_size :])

    for split in splits:
        np.random.shuffle(split)
    if any(not split for split in splits):
        raise ValueError(
            "The stratified split produced an empty set; provide more images "
            "from both classes."
        )
    return tuple(splits)


def create_dataloaders(
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    test_records: Sequence[ImageRecord],
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    train_transform, validation_transform = define_transforms()

    def make_dataset(records, transform):
        paths, labels = zip(*records)
        return GlassVialDataset(paths, labels, transform)

    return (
        DataLoader(
            make_dataset(train_records, train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        DataLoader(
            make_dataset(val_records, validation_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        DataLoader(
            make_dataset(test_records, validation_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    )


def calculate_image_statistics(data_loader: DataLoader):
    """Calculate average per-image channel statistics as in the notebook."""
    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        batch_images = images.numpy()
        batch_size = batch_images.shape[0]
        batch_images = batch_images.transpose(0, 2, 3, 1).reshape(
            batch_size,
            -1,
            3,
        )
        total_mean += batch_images.mean(axis=1).sum(axis=0)
        total_std += batch_images.std(axis=1).sum(axis=0)
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot calculate image statistics for an empty dataset.")
    return {
        "mean": (total_mean / total_samples).tolist(),
        "std": (total_std / total_samples).tolist(),
        "total_samples": total_samples,
    }


def _macro_f1(labels, predictions) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def train_cnn_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device=None,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """Train and early-stop on validation macro-F1 as in the notebook."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    if num_positive == 0 or num_negative == 0:
        raise ValueError("Training data must contain examples from both classes.")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [num_negative / num_positive],
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
        "val_f1_macro": [],
        "best_val_f1_macro": 0.0,
        "best_model_state": None,
    }
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_labels_epoch = []
        train_predictions = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_predictions.extend(
                (torch.sigmoid(logits) >= DEFAULT_THRESHOLD).view(-1).cpu().numpy()
            )
            train_labels_epoch.extend(labels.view(-1).cpu().numpy())

        model.eval()
        val_losses = []
        val_labels_epoch = []
        val_predictions = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)
                logits = model(images)
                val_losses.append(criterion(logits, labels).item())
                val_predictions.extend(
                    (torch.sigmoid(logits) >= DEFAULT_THRESHOLD)
                    .view(-1)
                    .cpu()
                    .numpy()
                )
                val_labels_epoch.extend(labels.view(-1).cpu().numpy())

        train_accuracy = float(
            accuracy_score(train_labels_epoch, train_predictions)
        )
        val_accuracy = float(accuracy_score(val_labels_epoch, val_predictions))
        val_macro_f1 = _macro_f1(val_labels_epoch, val_predictions)
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_f1"].append(
            float(
                f1_score(
                    train_labels_epoch,
                    train_predictions,
                    average="binary",
                    zero_division=0,
                )
            )
        )
        history["val_f1"].append(
            float(
                f1_score(
                    val_labels_epoch,
                    val_predictions,
                    average="binary",
                    zero_division=0,
                )
            )
        )
        history["val_f1_macro"].append(val_macro_f1)
        scheduler.step(val_macro_f1)

        if val_macro_f1 > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = val_macro_f1
            history["best_model_state"] = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 2:
                break

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={history['train_loss'][-1]:.4f}, "
            f"val_loss={history['val_loss'][-1]:.4f}, "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    if history["best_model_state"] is not None:
        model.load_state_dict(history["best_model_state"])
    return model, history


def tune_threshold(
    model: SimpleCNN,
    val_loader: DataLoader,
    device,
) -> float:
    """Select the validation threshold with the best macro-F1."""
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in val_loader:
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).view(-1).cpu().numpy())
            labels.extend(batch_labels.view(-1).cpu().numpy())

    if not labels:
        raise ValueError("Cannot tune a threshold on an empty validation set.")
    best_threshold = DEFAULT_THRESHOLD
    best_f1 = 0.0
    for threshold in np.arange(0.025, 0.976, 0.025):
        threshold = round(float(threshold), 3)
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        macro_f1 = _macro_f1(labels, predictions)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = threshold
    return best_threshold


def evaluate_model(
    model: SimpleCNN,
    test_loader: DataLoader,
    threshold: float,
    device,
):
    """Return the test metrics reported by the original notebook."""
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).view(-1).cpu().numpy())
            labels.extend(batch_labels.view(-1).cpu().numpy())

    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_score": float(
            f1_score(labels, predictions, average="binary", zero_division=0)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def compute_baseline_statistics(
    model: SimpleCNN,
    data_loader: DataLoader,
    device,
    threshold: float,
):
    """Compute validation-set image, class, prediction, confidence, and fc1 stats."""
    model.eval()
    image_means = []
    image_stds = []
    class_counts = {0: 0, 1: 0}
    prediction_counts = {0: 0, 1: 0}
    confidences = []
    embeddings = []

    feature_layer = model.fc1

    def capture_embedding(_module, _inputs, output):
        embeddings.append(output.detach().cpu().numpy())

    hook = feature_layer.register_forward_hook(capture_embedding)
    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                logits = model(images)
                probabilities = torch.sigmoid(logits).view(-1).cpu().numpy()
                predictions = (probabilities >= threshold).astype(int)

                image_means.append(images.mean(dim=(0, 2, 3)).cpu().numpy())
                image_stds.append(images.std(dim=(0, 2, 3)).cpu().numpy())
                for label in labels.view(-1).numpy():
                    class_counts[int(label)] += 1
                for prediction in predictions:
                    prediction_counts[int(prediction)] += 1
                confidences.extend(probabilities)
    finally:
        hook.remove()

    if not confidences:
        raise ValueError("Cannot calculate a drift baseline from an empty dataset.")
    means = np.asarray(image_means)
    stds = np.asarray(image_stds)
    result = {
        "image_statistics": {
            "mean_r": float(np.mean(means[:, 0])),
            "mean_g": float(np.mean(means[:, 1])),
            "mean_b": float(np.mean(means[:, 2])),
            "std_r": float(np.mean(stds[:, 0])),
            "std_g": float(np.mean(stds[:, 1])),
            "std_b": float(np.mean(stds[:, 2])),
        },
        "class_distribution": {
            "class_0_count": class_counts[0],
            "class_1_count": class_counts[1],
            "total": sum(class_counts.values()),
        },
        "prediction_distribution": {
            "pred_0_count": prediction_counts[0],
            "pred_1_count": prediction_counts[1],
            "total": sum(prediction_counts.values()),
        },
        "confidence_statistics": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        },
        "embedding_statistics": {
            "mean": [],
            "std": [],
        },
    }
    if embeddings:
        all_embeddings = np.concatenate(embeddings, axis=0)
        result["embedding_statistics"] = {
            "mean": np.mean(all_embeddings, axis=0).tolist(),
            "std": np.std(all_embeddings, axis=0).tolist(),
            "shape": int(all_embeddings.shape[1]),
        }
    return result


def save_checkpoint(
    model: SimpleCNN,
    path: Path,
    threshold: float,
) -> None:
    """Save the same checkpoint keys consumed by the deployment notebook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": SimpleCNN.__name__,
            "input_shape": (3, 64, 64),
            "output_shape": (1,),
            "threshold": float(threshold),
            "num_classes": 2,
        },
        path,
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/images/metadata.csv"),
        help="CSV containing filename and binary class columns.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/images/raw"),
        help="Directory containing images listed in --metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vial_cnn_model.pth"),
        help="Output path for the PyTorch checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_image_dataset(args.metadata, args.images)
    train_records, val_records, test_records = split_dataset(
        records,
        random_seed=args.seed,
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_records,
        val_records,
        test_records,
        batch_size=args.batch_size,
    )
    _write_json(
        args.output.parent / "class_names.json",
        {0: "good", 1: "defective"},
    )
    _write_json(
        args.output.parent / "image_statistics.json",
        calculate_image_statistics(train_loader),
    )
    model, _history = train_cnn_model(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        random_seed=args.seed,
    )
    threshold = tune_threshold(model, val_loader, device=device)
    results = evaluate_model(model, test_loader, threshold, device)
    baseline = compute_baseline_statistics(
        model,
        val_loader,
        device=device,
        threshold=threshold,
    )

    save_checkpoint(model, args.output, threshold)
    _write_json(args.output.parent / "model_results.json", results)
    _write_json(args.output.parent / "drift_baseline_stats.json", baseline)
    print(f"Model checkpoint saved to {args.output}")
    print(f"Test metrics: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
