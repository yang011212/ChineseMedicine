import os

from pytorch_segmentation.data_utils.data_loader import create_data_loader
from pytorch_segmentation.models.unet import resnet50_unet
from pytorch_segmentation.train import train


def main():
    root = os.path.join(os.getcwd(), "tongue_data")
    train_images = os.path.join(root, "train_img")
    train_labels = os.path.join(root, "train_label")
    val_images = os.path.join(root, "test_img")
    val_labels = os.path.join(root, "test_label")

    n_classes = 2
    input_height, input_width = 576, 768
    batch_size = 2
    epochs = 30
    lr = 1e-4

    train_loader = create_data_loader(
        images_path=train_images,
        annotations_path=train_labels,
        batch_size=batch_size,
        n_classes=n_classes,
        input_height=input_height,
        input_width=input_width,
        output_height=input_height,
        output_width=input_width,
        shuffle=True,
        num_workers=0,
        augment=True,
    )

    val_loader = create_data_loader(
        images_path=val_images,
        annotations_path=val_labels,
        batch_size=batch_size,
        n_classes=n_classes,
        input_height=input_height,
        input_width=input_width,
        output_height=input_height,
        output_width=input_width,
        shuffle=False,
        num_workers=0,
        augment=False,
    )

    model = resnet50_unet(n_classes=n_classes, input_height=input_height, input_width=input_width)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weights_dir="weights",
        model_prefix="tinyunet_pt",
    )


if __name__ == "__main__":
    main()
