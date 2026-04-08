import numpy as np


def merge_feature(features: np.ndarray, score: np.ndarray) -> float:
    """
    根据经验权重融合特征：sum(features * score)
    """
    features = np.asarray(features, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if features.shape != score.shape:
        raise ValueError(f"Shape mismatch: features{features.shape} vs score{score.shape}")
    return float(np.sum(np.multiply(features, score)))


def scaler_feature(x: np.ndarray) -> np.ndarray:
    """
    归一化到 0~1
    """
    x = np.asarray(x, dtype=np.float64)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def merge_region(region_scores: dict, region_ratios: dict) -> float:
    """
    根据区域占比融合各区域得分
    """
    total = 0.0
    for k, s in region_scores.items():
        w = float(region_ratios.get(k, 0.0))
        total += float(s) * w
    return float(total)


def health_score(x: float) -> float:
    """
    将得分映射到 [-1, 1]，越接近 0 越好
    这里使用 tanh 做平滑压缩。
    """
    return float(np.tanh(x))