import os
import cv2
import numpy as np
import gradio as gr
import json
import uuid
import requests
import base64
from PIL import Image, ImageDraw, ImageFont

def safe_imread(path, flag=1):
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0: return None
    return cv2.imdecode(data, flag)

from run import find_tongue, analysis, _judge_status

# 五大脏器区域颜色定义 (BGR格式)
REGION_COLORS = {
    "lung": (0, 0, 255),        # 心肺: 红色
    "spleen": (0, 255, 255),    # 脾: 黄色
    "kidney": (0, 0, 0),        # 肾: 黑色
    "liver_left": (0, 255, 0),  # 左肝: 绿色
    "liver_right": (0, 255, 0), # 右肝: 绿色
}
REGION_COLORS["kidney"] = (128, 0, 128)  # 使用紫色代替纯黑便于展示

def overlay_mask(img_bgr, mask_np, color, alpha=0.5):
    """
    将 mask 叠加到 img 上
    img_bgr: 原始 BGR 图像
    mask_np: 0/1 掩码
    color: 叠加颜色 (BGR)
    alpha: 透明度
    """
    h, w = img_bgr.shape[:2]
    if mask_np.shape[:2] != (h, w):
        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
    color_mask = np.zeros_like(img_bgr)
    color_mask[mask_np > 0] = color
    
    overlay = img_bgr.copy()
    overlay[mask_np > 0] = cv2.addWeighted(img_bgr, 1 - alpha, color_mask, alpha, 0)[mask_np > 0]
    return overlay

def put_chinese_text_center(img_bgr, text, mask_np, text_color=(255, 255, 255)):
    """
    在掩码的几何中心绘制中文
    """
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return img_bgr
    
    # 计算重心
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    
    # 加载系统字体
    try:
        font = ImageFont.truetype("msyh.ttc", 20)  # 微软雅黑
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", 20) # 黑体
        except:
            font = ImageFont.load_default()
            
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 获取文字大小以居中
    try:
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except:
        tw, th = len(text) * 20, 20
        
    tx = cx - tw // 2
    ty = cy - th // 2
    
    # 绘制黑色描边，增加辨识度
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
        draw.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0))
    # 绘制主体文字
    draw.text((tx, ty), text, font=font, fill=text_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_advise(scores, status):
    """
    如果环境变量中有 NVIDIA_API_KEY，则调用 API 生成大模型建议。
    否则使用本地中医规则模版。
    """
    prompt = f"""
这是一份基于图像的中医舌诊自动检测结果。
各项健康分数（范围通常在-1到1左右，越接近0代表越健康）：
整体: {scores.get('healthy', 0):.3f}
心肺: {scores.get('lung', 0):.3f}
脾: {scores.get('spleen', 0):.3f}
肾: {scores.get('kidney', 0):.3f}
左肝: {scores.get('liver_left', 0):.3f}
右肝: {scores.get('liver_right', 0):.3f}

系统初步判断状态为: 【{status}】

请你扮演一位专业的中医，根据以上检测结果，给出简短且结构化的中医建议（不超过200字）。
要求必须包含以下三点：
1. 饮食建议：应多吃什么，少吃什么。
2. 生活习惯：作息、运动或情绪调节建议。
3. 就医建议：是否需要线下寻求专业医生面诊。
    """
    
    api_key = os.environ.get("NVIDIA_API_KEY", "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").strip()
    if not api_key:
        # Fallback local rules
        advice = f"根据系统检测，您的身体状态为：【{status}】。\n"
        if status == "健康":
            advice += "\n1. 饮食建议：日常继续保持清淡饮食，营养均衡即可。\n2. 生活习惯：保持良好的作息时间，适度运动。\n3. 就医建议：当前指标良好，无需就医。"
        else:
            advice += "\n1. 饮食建议：针对当前评估出的体质偏颇，建议饮食以平性、易消化为主，避免大寒大热、辛辣油腻及过度进补。\n2. 生活习惯：注意规律作息，避免熬夜；可根据具体体质选择八段锦等舒缓运动调理气血；同时注意纾解情绪，避免过度焦虑。\n3. 就医建议：本系统检测为初步参考；若您在日常生活中经常感到对应的身体不适（如乏力、易怒、胀痛等），建议及时前往中医院进行专业面诊与针对性辩证调理。"
        return advice
    else:
        # NVIDIA API call (Standard OpenAI format)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta/llama-3.1-405b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.7
        }
        try:
            resp = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", json=data, headers=headers, timeout=15)
            if resp.status_code == 200:
                rj = resp.json()
                return rj["choices"][0]["message"]["content"]
            else:
                return f"调用 NVIDIA API 失败，状态码: {resp.status_code}\n信息: {resp.text}"
        except Exception as e:
            return f"调用 API 时发生异常: {str(e)}"

def process_image(img_path):
    if img_path is None:
        yield None, None, "", "", None, ""
        return
    
    yield None, None, "开始舌体检测与整体分割...", "", None, "检测中..."
    
    # 1. 检测与分割
    try:
        res = find_tongue(img_path)
    except Exception as e:
        yield None, None, f"图像处理错误: {str(e)}", "", None, ""
        return

    if res.get("code", -1) != 0:
        yield None, None, f"检测失败: {res.get('msg', '未知错误')}", "", None, ""
        return

    time_cost = res.get("time_consuming", 0.0)
    time_str = f"约 {time_cost:.3f} 秒"

    # 原图
    original_img = safe_imread(img_path, 1)
    if original_img is None:
        yield None, None, "读取图像文件失败", "", None, ""
        return
        
    mask = res["mask"]
    
    # 统一尺寸：确保原图和推理返回的 mask 尺寸一致（如 576x768）
    h_m, w_m = mask.shape[:2]
    if original_img.shape[:2] != (h_m, w_m):
        original_img = cv2.resize(original_img, (w_m, h_m))
        
    regions = res.get("regions", {})
    
    # 获取裁剪区域 bounding box
    # res["box"] 是 (x, y, w, h)
    box = res.get("box")
    if box is None or len(box) != 4 or box[2] == 0 or box[3] == 0:
        # fallback to full image
        x, y, w, h = 0, 0, original_img.shape[1], original_img.shape[0]
    else:
        x, y, w, h = box

    # 裁剪
    cropped_img = original_img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    # 中间图：裁剪并叠加绿色舌体蒙版
    # 1 -> 舌体
    mid_img_bgr = cropped_img.copy()
    mid_img_bgr[cropped_mask == 0] = (0, 0, 0) # Remove background
    mid_img_rgb = cv2.cvtColor(mid_img_bgr, cv2.COLOR_BGR2RGB)
    
    # ---- 步骤 1：产出第一阶段结果（分割图片和推理速度） ----
    yield mid_img_rgb, None, "正在进行脏器区域分析...", "", None, time_str
    
    # 右侧图：五大脏器区域划分
    right_img_bgr = mid_img_bgr.copy()
    
    organ_names_map = {
        "kidney": "肾",
        "spleen": "脾胃",
        "lung": "心肺",
        "liver_left": "左肝",
        "liver_right": "右肝"
    }
    
    if regions:
        # 需要把 global image 坐标的 mask 同样裁剪到 roi
        for organ, r_var in regions.items():
            if organ in ["tongue", "bbox"]:
                continue
            if organ in REGION_COLORS:
                organ_mask = r_var[y:y+h, x:x+w]
                right_img_bgr = overlay_mask(right_img_bgr, organ_mask, color=REGION_COLORS[organ], alpha=0.5)
                
                # 为该脏器区域中心添加中文名称标识
                if organ in organ_names_map:
                    right_img_bgr = put_chinese_text_center(right_img_bgr, organ_names_map[organ], organ_mask, text_color=(255, 255, 255))

    right_img_rgb = cv2.cvtColor(right_img_bgr, cv2.COLOR_BGR2RGB)

    # 2. 健康得分计算 (传入一个随机 user_id 避免冲突)
    uid = str(uuid.uuid4())[:8]
    scores = analysis(img_path, user_id=uid)
    status = _judge_status(scores)
    
    scores_out = {
        "状态": status,
        "整体得分 (healthy)": round(scores.get("healthy", 0.0), 4),
        "心肺得分 (lung)": round(scores.get("lung", 0.0), 4),
        "脾胃得分 (spleen)": round(scores.get("spleen", 0.0), 4),
        "肾脏得分 (kidney)": round(scores.get("kidney", 0.0), 4),
        "肝脏得分 (liver)": round(0.5 * (scores.get("liver_left", 0.0) + scores.get("liver_right", 0.0)), 4)
    }
    
    results_json_str = json.dumps(scores_out, ensure_ascii=False, indent=4)
    
    # ---- 步骤 2：产出区域分割结果和诊断结果，等待大模型生成中医建议 ----
    yield mid_img_rgb, right_img_rgb, results_json_str, "正在请求大语言模型生成中医建议，请稍候...", None, time_str
    
    # 3. 中医建议 (LLM / 模板)
    advice_text = generate_advise(scores, status)

    # 4. 生成导出文件
    # 将结果写到一个临时 markdown / json 文件中供下载
    export_data = {
        "检测结果": scores_out,
        "中医建议": advice_text
    }
    export_filename = f"report_{uid}.json"
    export_path = os.path.join(os.getcwd(), export_filename)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    # ---- 步骤 3：最终产出所有结果 ----
    yield mid_img_rgb, right_img_rgb, results_json_str, advice_text, export_path, time_str

# ========================
# 构建 Gradio UI
# ========================
bg_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ui", "backpic.png"))
if os.path.isfile(bg_image_path):
    with open(bg_image_path, "rb") as f:
        bg_image_base64 = base64.b64encode(f.read()).decode("utf-8")
    bg_css_value = f"url('data:image/png;base64,{bg_image_base64}')"
else:
    bg_css_value = "none"

css = f"""
.gradio-container {{
    background-image: {bg_css_value};
    background-color: #FFF9DB;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}
.sticky-header {{
    position: sticky;
    top: 0;
    z-index: 1000;
    background-color: transparent !important;
    padding: 15px 0;
    border-bottom: 1px solid var(--border-color-primary, #e5e7eb);
    margin-bottom: 20px;
    text-align: center;
    border-radius: 10px;
    backdrop-filter: none;
}}
.header-title {{ font-size: 2.2em; font-weight: bold; margin: 0; }}
.subtitle {{ color: gray; font-size: 14px; margin-top: 5px; margin-bottom: 0px; }}
"""

with gr.Blocks(title="AI中医舌诊", css=css) as demo:
    gr.HTML("""
    <div class="sticky-header">
        <div class="header-title">AI中医舌诊</div>
        <div class="subtitle">使用Tiny-Unet训练，best IoU = 0.8655</div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="上传图片")
            time_text = gr.Textbox(label="单张分割推理速度", interactive=False, lines=1)
        with gr.Column():
            img_mid = gr.Image(label="分割结果", interactive=False)
        with gr.Column():
            img_right = gr.Image(label="脏器区域", interactive=False)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 检测结果")
            res_text = gr.Textbox(label="各项指标得分及状态", lines=10, interactive=False)
        with gr.Column():
            gr.Markdown("### 中医建议")
            advice_text = gr.Textbox(label="调用meta/llama-3.1-405b-instruct模型生成", lines=10, interactive=False)
            
    with gr.Row():
        export_file = gr.File(label="检测结果文件", interactive=False, visible=False)
        btn_export = gr.Button("导出所有结果", variant="primary")
        
    # 定义交互逻辑
    # 上传图片或改变图片时触发推断
    img_input.change(
        fn=process_image,
        inputs=[img_input],
        outputs=[img_mid, img_right, res_text, advice_text, export_file, time_text]
    )

    # 点击导出时，显示已经生成好的 file (Gradio v3/v4 可以直接把之前更新的 export_file 显示出来)
    def trigger_export(path):
        if path:
            return gr.update(visible=True, value=path)
        return gr.update(visible=False)

    btn_export.click(fn=trigger_export, inputs=[export_file], outputs=[export_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, allowed_paths=[os.path.abspath(os.path.join(os.path.dirname(__file__), "ui"))])
