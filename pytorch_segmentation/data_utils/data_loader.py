import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import six

DATA_LOADER_SEED = 0
random.seed(DATA_LOADER_SEED) 
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

class DataLoaderError(Exception):  # 抛出异常
    pass


def _imread_unicode(path, flag):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flag)


def _augment_image_and_mask(image, mask):
    # 水平翻转（图像与标签同步）
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    h, w = image.shape[:2]

    # 小角度随机旋转（图像与标签同步）
    if random.random() < 0.5:
        angle = random.uniform(-10.0, 10.0)
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask,
            mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # 随机缩放后裁剪回原大小（图像与标签同步）
    if random.random() < 0.4:
        scale = random.uniform(0.9, 1.1)
        new_w = max(2, int(w * scale))
        new_h = max(2, int(h * scale))
        image_s = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if scale >= 1.0:
            x0 = random.randint(0, new_w - w)
            y0 = random.randint(0, new_h - h)
            image = image_s[y0:y0 + h, x0:x0 + w]
            mask = mask_s[y0:y0 + h, x0:x0 + w]
        else:
            pad_x = w - new_w
            pad_y = h - new_h
            left = random.randint(0, pad_x)
            right = pad_x - left
            top = random.randint(0, pad_y)
            bottom = pad_y - top
            image = cv2.copyMakeBorder(
                image_s, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
            )
            mask = cv2.copyMakeBorder(
                mask_s, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0
            )

    # 亮度和对比度扰动（仅作用于图像）
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)   # 对比度
        beta = random.uniform(-20, 20)       # 亮度
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 轻微高斯噪声（仅作用于图像）
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, mask


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):

    # 可接受的图像和分割图格式（文件后缀）
    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp", '.jpg']

    image_files = []           # 存储图像文件信息 (文件名, 扩展名, 完整路径)
    segmentation_files = {}    # 存储分割文件信息，键为文件名，值为(扩展名, 完整路径)

    # 遍历图像目录，筛选出符合格式的图像文件
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    # 遍历分割图目录，筛选出符合格式的标签文件，并以文件名为键建立索引
    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)

            # 若有重复的标签文件（文件名相同），抛出异常
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []  # 存储图像-分割图配对路径

    # 对图像进行遍历，查找是否有对应的标签
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            # 找到对应标签，加入返回结果
            return_value.append((image_full_path,
                                 segmentation_files[image_file][1]))
        elif ignore_non_matching:
            # 未找到标签但选择忽略，跳过
            continue
        else:
            # 未找到标签且不忽略，抛出异常
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value  # 返回图像与标签配对路径列表

def get_image_array(image_input, width, height, imgNorm="sub_mean"):

    # 处理 numpy 数组输入（图像已加载）
    if type(image_input) is np.ndarray:
        img = image_input

    # 处理字符串路径输入（从文件读取图像）
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = _imread_unicode(image_input, 1)  # 以 BGR 格式读取图像（彩色）
        if img is None:
            raise DataLoaderError("get_image_array: failed to read image {0}"
                                  .format(image_input))

    # 输入类型不支持时抛出异常
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    # 图像归一化处理
    if imgNorm == "sub_mean":
        # 调整图像大小到指定的宽高
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)

        # 减去 ImageNet 数据集的均值，用于与预训练模型保持一致
        img[:, :, 0] -= 103.939  # B通道
        img[:, :, 1] -= 116.779  # G通道
        img[:, :, 2] -= 123.68   # R通道

        # 将 BGR 格式转换为 RGB（OpenCV 默认读取为 BGR）
        img = img[:, :, ::-1]

    # 将图像从 HWC（高度、宽度、通道）转为 CHW（通道、高度、宽度）格式
    img = img.transpose(2, 0, 1)

    # 保证数组在内存中是连续的（避免后续模型运行时报错）
    img = np.ascontiguousarray(img)

    return img

def get_segmentation_array(image_input, nClasses, width, height):

    # 处理不同类型的输入（字符串路径或 numpy 数组）
    if type(image_input) is np.ndarray:
        img = image_input  # 已经是图像数组，直接使用
    elif isinstance(image_input, six.string_types):  # 字符串类型路径
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = _imread_unicode(image_input, 0)  # 以灰度模式读取标签图像（每个像素表示一个类别）
        if img is None:
            raise DataLoaderError("get_segmentation_array: failed to read label {0}"
                                  .format(image_input))
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Unsupported input type: {}".format(type(image_input)))

    # 调整标签图大小，使用"最近邻"插值以避免类别值被插值成浮点数
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    # 创建 one-hot 编码的标签数组，shape 为 (类别数, 高, 宽)
    seg_labels = np.zeros((nClasses, height, width), dtype=np.float32)

    # 将每个像素的类别编号转化为 one-hot 编码
    for c in range(nClasses):
        # 如果 img 中某像素值等于当前类 c，对应位置标记为 1
        seg_labels[c, :, :] = (img == c).astype(np.float32)

    return seg_labels

class SegmentationDataset(Dataset):
    """PyTorch 语义分割数据集类"""

    def __init__(self, images_path, annotations_path, n_classes,
                 input_height, input_width, output_height, output_width, augment=False):

        self.n_classes = n_classes
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.augment = augment

        # 获取图像与标签的配对路径列表
        self.image_seg_pairs = get_pairs_from_paths(images_path, annotations_path)

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.image_seg_pairs)

    def __getitem__(self, idx):
        """
        获取第 idx 个样本，包括图像和其对应的标签

        参数：
            idx: 索引

        返回：
            image: 处理后的图像张量，shape = (3, input_height, input_width)
            segmentation: one-hot 编码标签张量，shape = (n_classes, output_height, output_width)
        """
        image_path, seg_path = self.image_seg_pairs[idx]

        image = _imread_unicode(image_path, 1)
        if image is None:
            raise DataLoaderError("failed to read image {0}".format(image_path))
        mask = _imread_unicode(seg_path, 0)
        if mask is None:
            raise DataLoaderError("failed to read label {0}".format(seg_path))

        if self.augment:
            image, mask = _augment_image_and_mask(image, mask)

        # 加载并预处理图像
        image = get_image_array(image, self.input_width, self.input_height)

        # 加载并预处理标签（one-hot 编码）
        segmentation = get_segmentation_array(mask, self.n_classes,
                                              self.output_width, self.output_height)

        # 转为 PyTorch tensor，并转换为 float 类型
        image = torch.from_numpy(image).float()
        segmentation = torch.from_numpy(segmentation).float()

        return image, segmentation
    
def create_data_loader(images_path, annotations_path, batch_size, n_classes,
                      input_height, input_width, output_height, output_width,
                      shuffle=True, num_workers=0, augment=False):

    # 创建自定义的语义分割数据集对象
    dataset = SegmentationDataset(images_path, annotations_path, n_classes,
                                   input_height, input_width, output_height, output_width, augment=augment)

    # 使用 PyTorch 提供的 DataLoader 包装数据集
    dataloader = DataLoader(
        dataset,               # 传入数据集
        batch_size=batch_size,  # 批量大小
        shuffle=shuffle,        # 是否打乱顺序
        num_workers=num_workers,# 加载进程数
        pin_memory=True         # 是否将数据加载到固定内存中（加速 GPU 训练）
    )

    return dataloader  # 返回可迭代的数据加载器    