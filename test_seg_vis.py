import os
import glob
import cv2

from pytorch_segmentation.predict import model_from_checkpoint_path, predict

def main():
    # 设定测试文件夹和输出文件夹
    # 默认优先找 'tongue_data/test_img'，如果没有就找 'example/'
    test_dir = os.path.join(os.getcwd(), "tongue_data", "test_img")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(os.getcwd(), "example")
    
    out_dir = os.path.join(os.getcwd(), "seg_output")
    os.makedirs(out_dir, exist_ok=True)
    
    # 查找所有图片格式
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        # 支持大小写后缀
        img_paths.extend(glob.glob(os.path.join(test_dir, ext.upper())))

    if not img_paths:
        print(f"❌ 没有在目录 {test_dir} 中找到任何测试图片。请放置一些图片或修改脚本中的 test_dir。")
        return
    
    print(f"✅ 找到 {len(img_paths)} 张测试图片。输出将被保存到: {out_dir}")

    print("🚀 正在加载模型权重...")
    model, meta = model_from_checkpoint_path(weights_dir="weights", model_prefix="tinyunet_pt")
    
    # 遍历推理测试
    for img_path in img_paths:
        name = os.path.basename(img_path)
        save_path = os.path.join(out_dir, name)
        
        try:
            print(f"⏳ 正在处理: {name} ...")
            # predict 返回掩码(pred)和可视化叠加图(vis)，内部已调用 cv2.addWeighted 半透明叠加
            pred, vis = predict(
                model=model,
                img_path=img_path,
                input_height=meta["input_height"],
                input_width=meta["input_width"],
                device=meta["device"],
                save_path=save_path
            )
            print(f"  👉 成功，已保存至: {save_path}")
        except Exception as e:
            print(f"  ❌ 处理 {name} 失败！错误: {e}")

    print("🎉 测试结束，您可以去 seg_output 文件夹查看叠加了分割色块的效果图。")

if __name__ == "__main__":
    main()
