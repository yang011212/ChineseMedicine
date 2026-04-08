import cv2
import numpy as np
from typing import Dict, Tuple

def get_tongue(mask: np.ndarray) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    mask: (H,W) 0/1 或 0/255
    return: (x,y,w,h), tongue_mask(0/1)
    """
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0), m
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)), m

def Bezier2(p0, p1, p2, n=50):
    """
    二阶贝塞尔曲线采样点
    p0,p1,p2: (x,y)
    """
    t = np.linspace(0, 1, n).reshape(-1, 1)
    p0 = np.array(p0).reshape(1, 2)
    p1 = np.array(p1).reshape(1, 2)
    p2 = np.array(p2).reshape(1, 2)
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return pts.astype(np.int32)

def viscera_split(mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    输入：舌体 mask (H,W)
    输出：五个区域 mask（0/1），以及 tongue 总 mask
    区域：lung(心肺), spleen(脾胃), kidney(肾), liver_left(左肝), liver_right(右肝)
    
    【终极中医分割逻辑】：
    - 中间脾胃被四条边包围，形成一个四边形，其四个顶点均精确坐落于真正的舌体边缘上。
    - 四条边：上边向下凸，下边向上凸，左右边向内凸。
    - 其余四周的剩余部分直接通过脾胃边界线上下切分，划归给肾、心肺、左肝和右肝。
    """
    (x, y, w, h), tongue = get_tongue(mask)
    H, W = tongue.shape[:2]
    out = {
        "tongue": tongue.copy(),
        "lung": np.zeros((H, W), np.uint8),
        "spleen": np.zeros((H, W), np.uint8),
        "kidney": np.zeros((H, W), np.uint8),
        "liver_left": np.zeros((H, W), np.uint8),
        "liver_right": np.zeros((H, W), np.uint8),
        "bbox": np.array([x, y, w, h], dtype=np.int32),
    }
    if w == 0 or h == 0 or tongue.sum() == 0:
        return out

    # 找到舌体每一行的最左和最右坐标，以确定准确的边缘点
    m_ys, m_xs = np.where(tongue > 0)
    
    def get_edge_x(target_y):
        target_y = int(np.clip(target_y, m_ys.min(), m_ys.max()))
        row_xs = m_xs[m_ys == target_y]
        if len(row_xs) > 0:
            return int(row_xs.min()), int(row_xs.max())
        return x, x + w

    # ==== 1. 确定四大顶点 (全在真实舌头边缘) ====
    # 我们仍用高度比例确定上下切割横面的所处 y 值
    y_top = y + int(0.15 * h)  # 上移至左上角和右上角的位置，使上方肾区呈现半圆
    y_bot = y + int(0.72 * h)  # 调大下边界比例，以缩小下部的心肺区域
    
    tl_x, tr_x = get_edge_x(y_top)  # Top-Left, Top-Right
    bl_x, br_x = get_edge_x(y_bot)  # Bottom-Left, Bottom-Right
    
    p_tl = (tl_x, y_top)
    p_tr = (tr_x, y_top)
    p_bl = (bl_x, y_bot)
    p_br = (br_x, y_bot)
    
    # ==== 2. 绘制四大边界曲线 ====
    # 上边(肾/脾胃)向下凸（y坐标增加）
    c_top = Bezier2(p_tl, (x + w // 2, y_top + int(0.18 * h)), p_tr, n=50)
    
    # 下边(脾胃/心肺)向上凸
    c_bot = Bezier2(p_bl, (x + w // 2, y_bot - int(0.18 * h)), p_br, n=50)
    
    # 左边(左肝/脾胃)向内凸（横向控制点设在舌宽度的35%处，扩张左肝）
    ctrl_left_x = x + int(0.35 * w)
    c_left = Bezier2(p_tl, (ctrl_left_x, (y_top + y_bot) // 2), p_bl, n=50)
    
    # 右边(右肝/脾胃)向内凸（横向控制点设在舌宽度的65%处，扩张右肝）
    ctrl_right_x = x + int(0.65 * w)
    c_right = Bezier2(p_tr, (ctrl_right_x, (y_top + y_bot) // 2), p_br, n=50)

    # ==== 3. 填充中央的脾胃 ====
    pts_top = c_top.reshape(-1, 2).tolist()
    pts_right = c_right.reshape(-1, 2).tolist()
    pts_bot = c_bot.reshape(-1, 2).tolist()[::-1]
    pts_left = c_left.reshape(-1, 2).tolist()[::-1]
    
    poly_spleen = pts_top + pts_right + pts_bot + pts_left
    cv2.fillPoly(out["spleen"], [np.array(poly_spleen, np.int32)], 1)
    # 确保不超出原舌头
    out["spleen"] = (out["spleen"] & tongue).astype(np.uint8)

    # ==== 4. 用几何分割器切分其余四个角，保证无缝无余缝 ====
    # 我们创建一个全高宽掩码辅助切割
    # 肾 (kidney)：处于 c_top 之上，且排除脾胃
    poly_k = pts_top + [(tr_x, y), (tl_x, y)]
    cv2.fillPoly(out["kidney"], [np.array(poly_k, np.int32)], 1)
    
    # 心肺 (lung)：处于 c_bot 之下，且排除脾胃
    poly_l = pts_bot[::-1] + [(br_x, y + h), (bl_x, y + h)]
    cv2.fillPoly(out["lung"], [np.array(poly_l, np.int32)], 1)
    
    # 左肝 (liver_left)：处于 c_left 左侧
    poly_ll = pts_left[::-1] + [(x, y_bot), (x, y_top)]
    cv2.fillPoly(out["liver_left"], [np.array(poly_ll, np.int32)], 1)
    
    # 右肝 (liver_right)：处于 c_right 右侧
    poly_lr = pts_right + [(x + w, y_bot), (x + w, y_top)]
    cv2.fillPoly(out["liver_right"], [np.array(poly_lr, np.int32)], 1)

    # ==== 5. 做最终修边与去重补漏 ====
    for k in ["lung", "kidney", "liver_left", "liver_right"]:
        # 通过与总舌面做交集过滤外部背景
        out[k] = (out[k] & tongue).astype(np.uint8)
        # 强制抠掉可能与脾胃重叠的交界像素，脾胃优先级最高
        out[k][out["spleen"] == 1] = 0

    return out