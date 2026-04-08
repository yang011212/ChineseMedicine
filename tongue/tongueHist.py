from typing import Dict, Tuple

import cv2
import numpy as np

from tongue.segmentation_tongue import seg_tongue


def getStatistics(hist: np.ndarray) -> Dict[str, float]:
    """
    根据 1D 直方图计算统计量：
    - peak: 峰值位置（最大频数的 bin index）
    - std_width: 标准差宽度（像素值的标准差）
    - mean: 均值
    - trunc_mean: 截断均值（去掉两端 10% 后的均值）
    """
    hist = hist.flatten().astype(np.float64)
    total = hist.sum() + 1e-12
    bins = np.arange(len(hist), dtype=np.float64)

    peak = float(np.argmax(hist))
    mean = float((hist * bins).sum() / total)
    var = float((hist * (bins - mean) ** 2).sum() / total)
    std_width = float(np.sqrt(max(var, 0.0)))

    # 截断均值：按累计分布裁掉 10% 和 90% 之外的区间
    cdf = np.cumsum(hist) / total
    lo = int(np.searchsorted(cdf, 0.10))
    hi = int(np.searchsorted(cdf, 0.90))
    if hi <= lo:
        trunc_mean = mean
    else:
        h2 = hist[lo : hi + 1]
        b2 = bins[lo : hi + 1]
        trunc_mean = float((h2 * b2).sum() / (h2.sum() + 1e-12))

    return {"peak": peak, "std_width": std_width, "mean": mean, "trunc_mean": trunc_mean}


def calcuVec(img_bgr: np.ndarray, region_mask01: np.ndarray) -> Dict[str, float]:
    """
    遍历多个颜色空间与通道，对区域内像素计算直方图统计量。
    输出扁平化特征字典，键名包含颜色空间/通道/统计量。
    """
    mask = (region_mask01 > 0).astype(np.uint8)
    if mask.sum() == 0:
        return {}

    spaces = {
        "BGR": img_bgr,
        "RGB": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "HLS": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS),
        "LAB": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB),
    }

    feats: Dict[str, float] = {}
    for sp_name, sp_img in spaces.items():
        for ch in range(3):
            # 256 bins for 8-bit channels
            hist = cv2.calcHist([sp_img], [ch], mask, [256], [0, 256])
            stats = getStatistics(hist)
            for k, v in stats.items():
                feats[f"{sp_name}_c{ch}_{k}"] = float(v)
    return feats


def getVec(img_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    计算：
    - tongue_feats：舌体整体特征（dict）
    - region_ratios：各区域像素占比（dict）
    - region_feats：各区域特征（dict of dict）
    """
    data = np.fromfile(img_path, dtype=np.uint8)
    img = None if data.size == 0 else cv2.imdecode(data, 1)
    
    if img is None:
        raise FileNotFoundError(img_path)

    regions, bbox, tongue_mask = seg_tongue(img_path)
    # 统一到与 mask 相同的分辨率进行特征提取（seg_tongue 内部已用模型输入分辨率）
    h, w = tongue_mask.shape[:2]
    img = cv2.resize(img, (w, h))

    tongue_area = float(tongue_mask.sum()) + 1e-12
    region_keys = ["lung", "spleen", "kidney", "liver_left", "liver_right"]

    region_ratios: Dict[str, float] = {}
    region_feats: Dict[str, Dict[str, float]] = {}

    for k in region_keys:
        mk = regions.get(k, np.zeros_like(tongue_mask))
        region_ratios[k] = float(mk.sum()) / tongue_area
        region_feats[k] = calcuVec(img, mk)

    tongue_feats = calcuVec(img, tongue_mask)
    return tongue_feats, region_ratios, region_feats