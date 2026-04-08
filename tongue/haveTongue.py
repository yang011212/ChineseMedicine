import time
from typing import Dict, Tuple

import cv2
import numpy as np

from tongue.segmentation_tongue import seg_tongue

# 亮度阈值（全图/舌体ROI）
DARK_MEAN = 50
BRIGHT_MEAN = 200

# 舌体面积占比阈值（过小：没拍到/太远；过大：裁剪不完整/太近）
MIN_TONGUE_RATIO = 0.08
MAX_TONGUE_RATIO = 0.75

def calcuAera(img_path: str) -> Tuple[Dict, np.ndarray]:
    """
    返回：
      info: dict，包含亮度、bbox、面积占比等
      tongue_mask: (H,W) 0/1
    """
    data = np.fromfile(img_path, dtype=np.uint8)
    img = None if data.size == 0 else cv2.imdecode(data, 1)
    
    if img is None:
        raise FileNotFoundError(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_all = float(gray.mean())

    regions, bbox, tongue_mask = seg_tongue(img_path)
    
    # 统一尺寸：确保灰度图和推理返回的 mask 尺寸一致
    h_m, w_m = tongue_mask.shape[:2]
    if gray.shape[:2] != (h_m, w_m):
        gray = cv2.resize(gray, (w_m, h_m))
        mean_all = float(gray.mean()) # 更新全图平均亮度

    x, y, w, h = bbox

    if w == 0 or h == 0 or tongue_mask.sum() == 0:
        return {
            "mean_all": mean_all,
            "mean_roi": None,
            "ratio": 0.0,
            "bbox": bbox,
            "has_tongue": False,
        }, tongue_mask

    roi = gray[y : y + h, x : x + w]
    mean_roi = float(roi.mean()) if roi.size > 0 else mean_all

    ratio = float(tongue_mask.sum()) / float(tongue_mask.size)

    return {
        "mean_all": mean_all,
        "mean_roi": mean_roi,
        "ratio": ratio,
        "bbox": bbox,
        "has_tongue": True,
        "regions": regions,  # 后续任务8可复用
    }, tongue_mask


# 设置权重目录和模型前缀
WEIGHTS_DIR = "weights"
MODEL_PREFIX = "tinyunet_pt"

def haveTongue(img_path: str) -> Dict:
    """
    返回格式与任务10兼容：
      code: 0 合格；非0 不合格
      msg: 说明
      box: 舌体bbox
      mask: 舌体mask(0/1)
      time_consuming: 用时
    """
    t0 = time.time()
    info, mask = calcuAera(img_path)

    if not info["has_tongue"]:
        return {
            "code": 1,
            "msg": "未检测到舌体",
            "box": info["bbox"],
            "mask": mask,
            "time_consuming": time.time() - t0,
        }

    # 亮度判断：全图 + ROI 双判断更稳
    mean_all = info["mean_all"]
    mean_roi = info["mean_roi"] if info["mean_roi"] is not None else mean_all

    if mean_all < DARK_MEAN and mean_roi < DARK_MEAN:
        return {
            "code": 2,
            "msg": "图像过暗",
            "box": info["bbox"],
            "mask": mask,
            "time_consuming": time.time() - t0,
        }
    if mean_all > BRIGHT_MEAN and mean_roi > BRIGHT_MEAN:
        return {
            "code": 3,
            "msg": "图像过亮",
            "box": info["bbox"],
            "mask": mask,
            "time_consuming": time.time() - t0,
        }

    ratio = info["ratio"]
    if ratio < MIN_TONGUE_RATIO:
        return {
            "code": 4,
            "msg": "舌体占比过小（未拍到/距离过远）",
            "box": info["bbox"],
            "mask": mask,
            "time_consuming": time.time() - t0,
        }
    if ratio > MAX_TONGUE_RATIO:
        return {
            "code": 5,
            "msg": "舌体占比过大（舌体可能不完整/距离过近）",
            "box": info["bbox"],
            "mask": mask,
            "time_consuming": time.time() - t0,
        }

    return {
        "code": 0,
        "msg": "舌部质量检测通过",
        "box": info["bbox"],
        "mask": mask,
        "regions": info.get("regions", {}),
        "time_consuming": time.time() - t0,
    }