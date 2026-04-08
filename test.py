import os

from pytorch_segmentation.data_utils.data_loader import get_pairs_from_paths
from pytorch_segmentation.predict import model_from_checkpoint_path, evaluate

def get_model():
    model, meta = model_from_checkpoint_path(weights_dir="weights", model_prefix="tinyunet_pt")
    return model, meta

def test_img(test_img_dir: str, test_label_dir: str):
    pairs = get_pairs_from_paths(test_img_dir, test_label_dir)
    model, meta = get_model()
    metrics = evaluate(
        model,
        pairs,
        n_classes=meta["n_classes"],
        input_height=meta["input_height"],
        input_width=meta["input_width"],
        device=meta["device"],
        save_dir="prediction",
    )
    print(metrics)

def main():
    root = os.path.join(os.getcwd(), "tongue_data")
    test_img_dir = os.path.join(root, "test_img")
    test_label_dir = os.path.join(root, "test_label")
    test_img(test_img_dir, test_label_dir)


if __name__ == "__main__":
    main()
