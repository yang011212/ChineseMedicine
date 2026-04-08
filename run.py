import os
import time
from typing import Tuple, Dict

import cv2
import numpy as np

from tongue.haveTongue import haveTongue
from ChineseMedicine_analysis import analysis_ChineseMedicine


MAX_SIDE = 768  # 过大图片先缩放，避免推理太慢


def rm_img(path: str):
    if os.path.exists(path):
        os.remove(path)


def save_img(src_path: str, dst_path: str, max_side: int = MAX_SIDE) -> Tuple[float, int, int]:
    """
    调整并保存图像，返回 (ratio, new_w, new_h)
    ratio>1 表示缩放过（原图更大）
    """
    rm_img(dst_path)
    data = np.fromfile(src_path, dtype=np.uint8)
    img = None if data.size == 0 else cv2.imdecode(data, 1)
    if img is None:
        raise FileNotFoundError(src_path)
    h, w = img.shape[:2]
    ratio = max(w, h) / float(max_side)
    if ratio > 1.0:
        new_w = int(w / ratio)
        new_h = int(h / ratio)
        img = cv2.resize(img, (new_w, new_h))
    else:
        new_w, new_h = w, h
        ratio = 1.0
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, img)
    return ratio, new_w, new_h


def find_tongue(img_path: str) -> Dict:
    """
    舌部质量检测 + 舌体mask输出
    """
    t0 = time.time()
    res = haveTongue(img_path)
    res["time_consuming"] = float(res.get("time_consuming", 0.0) + (time.time() - t0))
    return res


def analysis(img_path: str, user_id: str) -> Dict:
    """
    计算健康得分，返回 dict（并写入 features/{user_id}/value.txt）
    """
    return analysis_ChineseMedicine(img_path, user_id=user_id)


def _judge_status(scores: Dict) -> str:
    """
    依据阈值输出所有检测到的健康状况（包含多个症状的合并）
    """
    healthy = scores.get("healthy", 0.0)
    kidney = scores.get("kidney", 0.0)
    spleen = scores.get("spleen", 0.0)
    lung = scores.get("lung", 0.0)
    liver = 0.5 * (scores.get("liver_left", 0.0) + scores.get("liver_right", 0.0))

    symptoms = []

    # --- 脾胃（舌中） ---
    if spleen < -0.5:
        symptoms.append("脾胃虚寒")
    elif spleen > 0.5:
        symptoms.append("脾胃湿热")

    # --- 肝胆（舌侧） ---
    if liver < -0.5:
        symptoms.append("肝气郁结")
    elif liver > 0.5:
        symptoms.append("肝火偏旺")

    # --- 心肺（舌尖） ---
    if lung < -0.5:
        symptoms.append("心肺气虚")
    elif lung > 0.5:
        symptoms.append("心肺蕴热")

    # --- 肾（舌根） ---
    if kidney < -0.5:
        symptoms.append("肾阳不足")
    elif kidney > 0.5:
        symptoms.append("肾阴虚火旺")

    # --- 全局倾向 ---
    if healthy > 0.5:
        symptoms.append("气血亏虚")
    elif healthy < -0.5:
        symptoms.append("痰湿较重")

    # 若没有任何症状被判定（或都在小阈值内），则认为是健康的
    if not symptoms:
        if abs(healthy) < 0.2 and all(abs(v) < 0.3 for v in [kidney, spleen, lung, liver]):
            return "健康"
        else:
            return "亚健康"

    return "、".join(symptoms)


def main(img_path: str, user_id: str = "001"):
    # 1) 质量检测
    res = find_tongue(img_path)
    if res["code"] != 0:
        print({"code": res["code"], "msg": res["msg"]})
        return

    # 2) 可视化：原图与 mask 拼接展示
    data = np.fromfile(img_path, dtype=np.uint8)
    img = None if data.size == 0 else cv2.imdecode(data, 1)
    mask = (res["mask"] > 0).astype(np.uint8) * 255
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask3 = cv2.resize(mask3, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    show = np.concatenate([img, mask3], axis=1)
    cv2.imshow("image | mask", show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3) 健康得分
    scores = analysis(img_path, user_id=user_id)
    status = _judge_status(scores)
    scores_out = {
        "healthy": scores.get("healthy", 0.0),
        "heart_lung": scores.get("lung", 0.0),
        "spleen": scores.get("spleen", 0.0),
        "kidney": scores.get("kidney", 0.0),
        "liver": 0.5 * (scores.get("liver_left", 0.0) + scores.get("liver_right", 0.0)),
        "status": status,
    }
    print(scores_out)


if __name__ == "__main__":
    # 示例：使用 example/ 下的图片（按你解压的路径调整）
    example_img = os.path.join(os.getcwd(), "example", "1.jpg")
    main(example_img, user_id="001")