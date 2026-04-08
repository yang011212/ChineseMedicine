import os
import json
from typing import Dict

from tongue.tongueHist import getVec
from analysis.main import Main


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def analysis_ChineseMedicine(img_path: str, user_id: str = "default") -> Dict:
    """
    计算整体及各区域健康值，并写入 features/{user_id}/value.txt
    返回 dict：包含 overall 与各区域分数
    """
    tongue_feats, region_ratios, region_feats = getVec(img_path)

    # 只用各区域特征做健康值（tongue_feats 可扩展加入）
    region_scores, total = Main(region_feats, region_ratios)

    out = {
        "healthy": float(total),
        "lung": float(region_scores.get("lung", 0.0)),
        "spleen": float(region_scores.get("spleen", 0.0)),
        "kidney": float(region_scores.get("kidney", 0.0)),
        "liver_left": float(region_scores.get("liver_left", 0.0)),
        "liver_right": float(region_scores.get("liver_right", 0.0)),
        "ratios": {k: float(v) for k, v in region_ratios.items()},
    }

    base = os.path.join(os.getcwd(), "features", str(user_id))
    _ensure_dir(base)
    value_path = os.path.join(base, "value.txt")

    # 读取历史（若存在）
    history = None
    if os.path.isfile(value_path):
        try:
            with open(value_path, "r", encoding="utf-8") as f:
                history = json.loads(f.read().strip() or "{}")
        except:
            history = None

    # 若存在历史，则做简单平均平滑（让二次诊断更稳）
    if isinstance(history, dict) and history:
        for k in ["healthy", "lung", "spleen", "kidney", "liver_left", "liver_right"]:
            out[k] = float(0.5 * history.get(k, 0.0) + 0.5 * out[k])

    with open(value_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))

    return out