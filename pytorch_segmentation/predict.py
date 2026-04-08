import os
import json
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import torch

from pytorch_segmentation.models.unet import resnet50_unet, tiny_unet
from pytorch_segmentation.models.resunet_plusplus import resunet_plusplus
from pytorch_segmentation.train import find_latest_checkpoint
from pytorch_segmentation.data_utils.data_loader import get_image_array

def _imread_unicode(path: str, flag: int):
    """Robust image read on Windows paths containing non-ASCII chars."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flag)


def _imwrite_unicode(path: str, image: np.ndarray) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True

def _load_config(weights_dir: str, model_prefix: str) -> Dict:
    cfg_path = os.path.join(weights_dir, f"{model_prefix}_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def model_from_checkpoint_path(
    weights_dir: str = "weights",
    model_prefix: str = "tinyunet_pt",
    device: Optional[str] = None,
    n_classes: int = 2,
    input_height: int = 576,
    input_width: int = 768,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = _load_config(weights_dir, model_prefix)
    n_classes = int(cfg.get("n_classes", n_classes))
    input_height = int(cfg.get("input_height", input_height))
    input_width = int(cfg.get("input_width", input_width))

    ckpt_path, ckpt_epoch = find_latest_checkpoint(weights_dir, prefix=model_prefix)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {weights_dir} (prefix={model_prefix})")

    mp = model_prefix.lower()
    if mp == "tinyunet_pt" or mp.startswith("tinyunet_pt"):
        model = tiny_unet(
            n_classes=n_classes,
            input_height=input_height,
            input_width=input_width,
            pretrained=True,
        )
    elif "tiny" in mp:
        model = tiny_unet(
            n_classes=n_classes,
            input_height=input_height,
            input_width=input_width,
            pretrained=False,
        )
    elif "resunetplusplus" in mp or "resnetplusplus" in mp:
        model = resunet_plusplus(n_classes=n_classes, input_height=input_height, input_width=input_width)
    else:
        model = resnet50_unet(n_classes=n_classes, input_height=input_height, input_width=input_width)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "checkpoint_path": ckpt_path,
        "checkpoint_epoch": ckpt_epoch,
        "n_classes": n_classes,
        "input_height": input_height,
        "input_width": input_width,
        "device": device,
    }
    return model, meta

def get_colored_segmentation_image(seg: np.ndarray, colors=None) -> np.ndarray:
    """
    seg: (H, W) 类别 id
    return: (H, W, 3) BGR 彩色图
    """
    if colors is None:
        colors = [
            (0, 0, 0),        # 0 背景
            (0, 255, 0),      # 1 舌体
            (0, 0, 255),
            (255, 0, 0),
        ]
    h, w = seg.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, c in enumerate(colors):
        out[seg == cls_id] = c
    return out

def visualize_segmentation(image_bgr: np.ndarray, seg: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color_seg = get_colored_segmentation_image(seg)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, color_seg, alpha, 0)
    return overlay

def predict(
    model,
    img_path: str,
    *,
    input_height: int,
    input_width: int,
    device: str,
    save_path: Optional[str] = None,
):
    img = _imread_unicode(img_path, 1)
    if img is None:
        raise FileNotFoundError(img_path)

    x = get_image_array(img, input_width, input_height)  # CHW, float32
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(x)  # (1,C,H,W)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    vis = visualize_segmentation(cv2.resize(img, (input_width, input_height)), pred)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _imwrite_unicode(save_path, vis)
    return pred, vis

def _fast_hist(pred: np.ndarray, label: np.ndarray, n_classes: int) -> np.ndarray:
    mask = (label >= 0) & (label < n_classes)
    hist = np.bincount(
        n_classes * label[mask].astype(int) + pred[mask].astype(int),
        minlength=n_classes ** 2,
    ).reshape(n_classes, n_classes)
    return hist


def evaluate(
    model,
    pairs,
    *,
    n_classes: int,
    input_height: int,
    input_width: int,
    device: str,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    pairs: [(img_path, label_path), ...]
    label_path: 灰度标签图（像素值为类别 id）
    """
    hist = np.zeros((n_classes, n_classes), dtype=np.float64)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for img_path, lab_path in pairs:
        pred, _ = predict(
            model,
            img_path,
            input_height=input_height,
            input_width=input_width,
            device=device,
            save_path=os.path.join(save_dir, os.path.basename(img_path)) if save_dir else None,
        )

        lab = _imread_unicode(lab_path, 0)
        if lab is None:
            raise FileNotFoundError(lab_path)
        lab = cv2.resize(lab, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
        hist += _fast_hist(pred, lab.astype(np.uint8), n_classes)

    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = float(np.nanmean(iou))
    return {"mIoU": miou, "IoU_per_class": iou.tolist()}