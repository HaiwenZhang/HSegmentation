import os
import cv2
import numpy as np
from torch.utils.data import Dataset

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

def _read_img_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
    return image

def _read_label(path):
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return label

def _read_voc_info(voc_dir, is_train=True):
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    return images

def _read_image_and_label(voc_dir, image_name):
    image_path = os.path.join(voc_dir, 'JPEGImages',  f'{image_name}.jpg')
    label_path = os.path.join(voc_dir, 'JPEGImages',  f'{image_name}.jpg')

    image = _read_img_rgb(image_path)
    label = _read_label(label_path)
    return image, label

def _voc_colormap2label():
    """构建从RGB到VOC类别索引的映射。"""
    colormap2label = np.zeros(256 ** 3, dtype=np.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def _convert_to_segmentation_mask(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引。"""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def _convert_to_segmentation_mask1(mask):
    """将VOC标签中的RGB值映射到它们的类别索引。"""
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
    for label_index, label in enumerate(VOC_COLORMAP):
        segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
    return segmentation_mask


class VOCSegDataset(Dataset):

    def __init__(self, root, is_train=True, transform=None):
        super(VOCSegDataset, self).__init__()

        self.is_train=is_train
        self.root = root
        self.images_info = _read_voc_info(self.root, self.is_train)
        self.length = len(self.images_info)
        self.transform = transform
        self.colormap2label = _voc_colormap2label()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_name = self.images_info[idx]
        image, label = _read_image_and_label(self.root, image_name)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = _convert_to_segmentation_mask1(mask)
        return image, mask



if __name__ == "__main__":
    
    voc_dir = "data/VOC2012"

    label_path = os.path.join(voc_dir,"SegmentationClass", "2007_000033.png")
    label = _read_img_rgb(label_path)
    colormap2label = _voc_colormap2label()
    mask = _convert_to_segmentation_mask(label, colormap2label)
    print(mask.shape)

    images = _read_voc_info(voc_dir, True)
    print(type(images))
     




