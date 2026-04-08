import os
import csv
from typing import Dict, Tuple

import numpy as np

from analysis.merge_features import merge_feature, scaler_feature, merge_region, health_score


REGIONS = ["lung", "spleen", "kidney", "liver_left", "liver_right"]


def _load_ls_csv(path: str):
    """
    读取 LS.csv（若存在），返回 dict:
      weights[region] = (feature_names, weights_array)
    兼容：csv首行表头；假设包含 region/name/weight 三列之一。
    """
    if not os.path.isfile(path):
        return None

    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 尝试推断列名
    keys = set(rows[0].keys()) if rows else set()
    region_key = "region" if "region" in keys else ("Region" if "Region" in keys else None)
    name_key = "name" if "name" in keys else ("feature" if "feature" in keys else ("Name" if "Name" in keys else None))
    w_key = "weight" if "weight" in keys else ("score" if "score" in keys else ("Weight" if "Weight" in keys else None))
    if region_key is None or name_key is None or w_key is None:
        return None

    weights = {r: {"names": [], "w": []} for r in REGIONS}
    for r in rows:
        reg = r.get(region_key)
        if reg not in weights:
            continue
        weights[reg]["names"].append(r.get(name_key))
        weights[reg]["w"].append(float(r.get(w_key, 0.0)))
    for reg in weights:
        weights[reg]["w"] = np.asarray(weights[reg]["w"], dtype=np.float64)
    return weights


def Main(features_by_region: Dict[str, Dict[str, float]], region_ratios: Dict[str, float], ls_csv_path: str = None):
    """
    输入：
      features_by_region: {region: {feat_name: feat_value}}
      region_ratios: {region: ratio}
    输出：
      region_scores: {region: score_in_-1_1}
      total_score: overall score_in_-1_1
    """
    if ls_csv_path is None:
        ls_csv_path = os.path.join(os.path.dirname(__file__), "LS.csv")
    ls = _load_ls_csv(ls_csv_path)

    region_raw = {}
    for reg in REGIONS:
        feats = features_by_region.get(reg, {})
        if not feats:
            region_raw[reg] = 0.0
            continue

        # 若有 LS.csv，则按其指定特征顺序取值并加权；否则用简单均值做 raw score
        if ls is not None and ls.get(reg) and len(ls[reg]["names"]) > 0:
            names = ls[reg]["names"]
            w = ls[reg]["w"]
            x = np.asarray([float(feats.get(n, 0.0)) for n in names], dtype=np.float64)
            x = scaler_feature(x)
            s = merge_feature(x, w)
        else:
            x = np.asarray(list(feats.values()), dtype=np.float64)
            x = scaler_feature(x)
            s = float(x.mean())  # baseline

        region_raw[reg] = s

    # 将 raw 区域分数映射到 [-1,1]
    region_scores = {k: health_score(v) for k, v in region_raw.items()}
    total = health_score(merge_region(region_scores, region_ratios))
    return region_scores, total