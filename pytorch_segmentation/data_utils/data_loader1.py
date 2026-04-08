import os
import random

import cv2
import numpy as np
import six
import torch
from torch.utils.data import DataLoader, Dataset


DATA_LOADER_SEED = 0
random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]


class DataLoaderError(Exception):
    pass


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    acceptable_image_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    acceptable_segmentation_formats = [".png", ".bmp", ".jpg"]

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        full_path = os.path.join(images_path, dir_entry)
        if os.path.isfile(full_path) and os.path.splitext(dir_entry)[1] in acceptable_image_formats:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension, full_path))

    for dir_entry in os.listdir(segs_path):
        full_path = os.path.join(segs_path, dir_entry)
        if os.path.isfile(full_path) and os.path.splitext(dir_entry)[1] in acceptable_segmentation_formats:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError(
                    "Segmentation file with filename {0} already exists and is ambiguous to resolve with path {1}."
                    " Please remove or rename the latter.".format(file_name, full_path)
                )
            segmentation_files[file_name] = (file_extension, full_path)

    return_value = []
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path, segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            raise DataLoaderError("No corresponding segmentation found for image {0}.".format(image_full_path))

    return return_value


def _apply_augmentations(image: np.ndarray, segmentation: np.ndarray):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        segmentation = cv2.flip(segmentation, 1)

    if random.random() < 0.15:
        image = cv2.flip(image, 0)
        segmentation = cv2.flip(segmentation, 0)

    if random.random() < 0.4:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-18, 18)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.2:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    if random.random() < 0.25:
        noise = np.random.normal(0, 6, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, segmentation


def get_image_array(image_input, width, height, imgNorm="sub_mean"):
    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}".format(str(type(image_input))))

    if imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]

    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


def get_segmentation_array(image_input, nClasses, width, height):
    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 0)
    else:
        raise DataLoaderError("get_segmentation_array: Unsupported input type: {}".format(type(image_input)))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    seg_labels = np.zeros((nClasses, height, width), dtype=np.float32)
    for c in range(nClasses):
        seg_labels[c, :, :] = (img == c).astype(np.float32)

    return seg_labels


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_path,
        annotations_path,
        n_classes,
        input_height,
        input_width,
        output_height,
        output_width,
        augment=False,
    ):
        self.n_classes = n_classes
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.augment = augment
        self.image_seg_pairs = get_pairs_from_paths(images_path, annotations_path)

    def __len__(self):
        return len(self.image_seg_pairs)

    def __getitem__(self, idx):
        image_path, seg_path = self.image_seg_pairs[idx]

        image = cv2.imread(image_path, 1)
        segmentation = cv2.imread(seg_path, 0)
        if image is None:
            raise DataLoaderError("Unable to read image: {}".format(image_path))
        if segmentation is None:
            raise DataLoaderError("Unable to read segmentation: {}".format(seg_path))

        if self.augment:
            image, segmentation = _apply_augmentations(image, segmentation)

        image = get_image_array(image, self.input_width, self.input_height)
        segmentation = get_segmentation_array(segmentation, self.n_classes, self.output_width, self.output_height)

        image = torch.from_numpy(image).float()
        segmentation = torch.from_numpy(segmentation).float()

        return image, segmentation


def create_data_loader(
    images_path,
    annotations_path,
    batch_size,
    n_classes,
    input_height,
    input_width,
    output_height,
    output_width,
    shuffle=True,
    num_workers=0,
    augment=False,
):
    dataset = SegmentationDataset(
        images_path,
        annotations_path,
        n_classes,
        input_height,
        input_width,
        output_height,
        output_width,
        augment=augment,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
