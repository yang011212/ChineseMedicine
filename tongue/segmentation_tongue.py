import os
import cv2
import numpy as np

from pytorch_segmentation.predict import model_from_checkpoint_path, predict
from tongue.tongue_segmentation.segmentation import get_tongue, viscera_split

WEIGHTS_DIR = "weights"
# MODEL_PREFIX = "tinyunet"
MODEL_PREFIX = "tinyunet_pt"

def seg_tongue(img_path: str):
    """
    返回：
      regions: dict，各区域mask（0/1）与 bbox
      tongue_bbox: (x,y,w,h)
      tongue_mask: (H,W) 0/1
    """
    model, meta = model_from_checkpoint_path(weights_dir=WEIGHTS_DIR, model_prefix=MODEL_PREFIX)
    pred, _ = predict(
        model,
        img_path,
        input_height=meta["input_height"],
        input_width=meta["input_width"],
        device=meta["device"],
        save_path=None,
    )

    # pred: (H,W) 类别id；默认 1 为舌体
    tongue_mask = (pred == 1).astype(np.uint8)
    bbox, _ = get_tongue(tongue_mask)
    regions = viscera_split(tongue_mask)
    return regions, bbox, tongue_mask
