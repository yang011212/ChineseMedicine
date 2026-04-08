import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_epoch_number_from_path(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"\.(\d+)\.(pt|pth)$", base)
    if not m:
        return None
    return int(m.group(1))


def find_latest_checkpoint(weights_dir: str, prefix: str = "tinyunet_pt") -> Tuple[Optional[str], int]:
    if not os.path.isdir(weights_dir):
        return None, -1

    best_path, best_epoch = None, -1
    for name in os.listdir(weights_dir):
        if not name.startswith(prefix + "."):
            continue
        full = os.path.join(weights_dir, name)
        if not os.path.isfile(full):
            continue
        epoch = get_epoch_number_from_path(full)
        if epoch is None:
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = full
    return best_path, best_epoch


def masked_categorical_crossentropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    if target.dim() == 4:
        target = target.argmax(dim=1)
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)


@dataclass
class CheckpointsCallback:
    weights_dir: str
    model_prefix: str = "tinyunet_pt"

    def __post_init__(self):
        os.makedirs(self.weights_dir, exist_ok=True)

    def save_config(self, config: Dict):
        path = os.path.join(self.weights_dir, f"{self.model_prefix}_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def save_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        path = os.path.join(self.weights_dir, f"{self.model_prefix}.{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
        return path


def _pixel_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    if target.dim() == 4:
        target = target.argmax(dim=1)
    pred = logits.argmax(dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return (pred[valid] == target[valid]).float().mean()


def _batch_hist(logits: torch.Tensor, target: torch.Tensor, n_classes: int, ignore_index: int = 255) -> np.ndarray:
    if target.dim() == 4:
        target = target.argmax(dim=1)
    pred = logits.argmax(dim=1)
    pred = pred.detach().cpu().numpy().astype(np.int64)
    target = target.detach().cpu().numpy().astype(np.int64)

    hist = np.zeros((n_classes, n_classes), dtype=np.float64)
    for p, t in zip(pred, target):
        valid = (t >= 0) & (t < n_classes) & (t != ignore_index)
        if not np.any(valid):
            continue
        hist += np.bincount(
            n_classes * t[valid].reshape(-1) + p[valid].reshape(-1),
            minlength=n_classes ** 2,
        ).reshape(n_classes, n_classes)
    return hist


def _miou_from_hist(hist: np.ndarray) -> float:
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    return float(np.nanmean(iou))


def _save_history_files(history: Dict[str, list], weights_dir: str, model_prefix: str):
    os.makedirs(weights_dir, exist_ok=True)
    csv_path = os.path.join(weights_dir, f"{model_prefix}_history.csv")
    json_path = os.path.join(weights_dir, f"{model_prefix}_history.json")
    txt_path = os.path.join(weights_dir, f"{model_prefix}_training_report.txt")
    loss_plot_path = os.path.join(weights_dir, f"{model_prefix}_loss_curve.png")
    iou_plot_path = os.path.join(weights_dir, f"{model_prefix}_iou_curve.png")

    keys = ["epoch", "train_loss", "train_iou", "val_loss", "val_iou"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({k: history[k][i] for k in keys})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("训练结果汇总\n")
        f.write("====================\n")
        for i in range(len(history["epoch"])):
            f.write(
                f"Epoch {history['epoch'][i]}: "
                f"train_loss={history['train_loss'][i]:.4f}, "
                f"train_iou={history['train_iou'][i]:.4f}, "
                f"val_loss={history['val_loss'][i]:.4f}, "
                f"val_iou={history['val_iou'][i]:.4f}\n"
            )

        best_val_iou_idx = int(np.argmax(history["val_iou"])) if history["val_iou"] else -1
        if best_val_iou_idx >= 0:
            f.write("\n最佳验证结果\n")
            f.write("--------------------\n")
            f.write(
                f"Epoch {history['epoch'][best_val_iou_idx]} | "
                f"val_loss={history['val_loss'][best_val_iou_idx]:.4f} | "
                f"val_iou={history['val_iou'][best_val_iou_idx]:.4f}\n"
            )

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_iou"], label="train_iou", linewidth=2)
    plt.plot(epochs, history["val_iou"], label="val_iou", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(iou_plot_path, dpi=200)
    plt.close()


def train(
    model: nn.Module,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 30,
    lr: float = 1e-4,
    device: Optional[str] = None,
    weights_dir: str = "weights",
    model_prefix: str = "tinyunet_pt",
    ignore_index: int = 255,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ckpt = CheckpointsCallback(weights_dir=weights_dir, model_prefix=model_prefix)
    n_classes = None

    latest_path, latest_epoch = find_latest_checkpoint(weights_dir, prefix=model_prefix)
    start_epoch = 0
    if latest_path is not None:
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("epoch", latest_epoch)) + 1

    config = {
        "model_prefix": model_prefix,
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "ignore_index": ignore_index,
    }
    ckpt.save_config(config)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_iou": [],
        "val_loss": [],
        "val_iou": [],
    }

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        train_hist = None

        for images, seg_onehot in train_loader:
            images = images.to(device)
            target = seg_onehot.argmax(dim=1).to(device)
            if n_classes is None:
                n_classes = int(seg_onehot.shape[1])

            optimizer.zero_grad()
            logits = model(images)
            loss = masked_categorical_crossentropy(logits, target, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_hist = _batch_hist(logits.detach(), target, n_classes, ignore_index=ignore_index)
            train_hist = batch_hist if train_hist is None else train_hist + batch_hist

        train_loss = running_loss / max(1, len(train_loader))
        train_iou = _miou_from_hist(train_hist if train_hist is not None else np.zeros((n_classes, n_classes)))

        val_loss = 0.0
        val_iou = 0.0
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            val_hist = None
            with torch.no_grad():
                for images, seg_onehot in val_loader:
                    images = images.to(device)
                    target = seg_onehot.argmax(dim=1).to(device)
                    logits = model(images)
                    loss = masked_categorical_crossentropy(logits, target, ignore_index=ignore_index)
                    v_loss += loss.item()
                    batch_hist = _batch_hist(logits, target, n_classes, ignore_index=ignore_index)
                    val_hist = batch_hist if val_hist is None else val_hist + batch_hist

            val_loss = v_loss / max(1, len(val_loader))
            val_iou = _miou_from_hist(val_hist if val_hist is not None else np.zeros((n_classes, n_classes)))

        saved_path = ckpt.save_epoch(model, optimizer, epoch)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_iou"].append(float(train_iou))
        history["val_loss"].append(float(val_loss))
        history["val_iou"].append(float(val_iou))
        _save_history_files(history, weights_dir, model_prefix)

        print(
            f"Epoch {epoch}/{epochs - 1} | "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} | "
            f"saved: {saved_path}"
        )

    return model, history
