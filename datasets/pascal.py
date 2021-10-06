import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision

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

def _read_image_and_label(voc_dir, image_name):
    image_path = os.path.join(voc_dir, 'JPEGImages',  f'{image_name}.jpg')
    label_path = os.path.join(voc_dir, 'JPEGImages',  f'{image_name}.jpg')

    image = _read_img_rgb(image_path)
    label = _read_img_rgb(label_path)
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


def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注。"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
       feature, label = _read_image_and_label(voc_dir=voc_dir, image_name=fname)
       features.append(feature)
       labels.append(label)
    return features, labels


class VOCSegDataset(Dataset):
    """一个用于加载VOC数据集的自定义数据集。"""

    def __init__(self, voc_dir, image_size, is_train=True, transform=None):

        self.crop_size = image_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = self.filter(features)
        self.labels = self.filter(labels)
        self.colormap2label = _voc_colormap2label()
        self.transform = transform
        self.to_tensor = T.Compose([
            T.ToTensor()
        ])
        print('keep ' + str(len(self.features)) + ' examples')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]


        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            mask = transformed["mask"]
        mask = _convert_to_segmentation_mask(mask, self.colormap2label)
        image = self.to_tensor(image)

        return image, mask

    def __len__(self):
        return len(self.features)




if __name__ == "__main__":
    
    voc_dir = "/Users/haiwen/AI/Datasets/PASCAL_VOC/VOC2012"

    label_path = os.path.join(voc_dir,"SegmentationClass", "2007_000033.png")
    # label = _read_img_rgb(label_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    print(np.unique(label))

    # colormap2label = _voc_colormap2label()
    # mask = _convert_to_segmentation_mask(label, colormap2label)
    # print(mask.shape)

     




