import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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


def label2image(pred, device):
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]

def _read_img_rgb(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return image

def _read_image_and_label(voc_dir, image_name):
    image_path = os.path.join(voc_dir, 'JPEGImages',  f'{image_name}.jpg')
    label_path = os.path.join(voc_dir, 'SegmentationClass',  f'{image_name}.png')

    image = _read_img_rgb(image_path)
    label = _read_img_rgb(label_path)
    return image, label

def _voc_colormap2label():

    colormap2label = np.zeros(256 ** 3, dtype=np.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def _convert_to_segmentation_mask(colormap, colormap2label):

    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def read_voc_images(voc_dir, is_train=True):

    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for _, fname in enumerate(images):
       feature, label = _read_image_and_label(voc_dir=voc_dir, image_name=fname)
       features.append(feature)
       labels.append(label)
    return features, labels

def label2image(pred, device):
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]


class VOCSegDataset(Dataset):

    def __init__(self, voc_dir, is_train=True, transform=None):

        self.features, self.labels = read_voc_images(voc_dir, is_train=is_train)
        self.colormap2label = _voc_colormap2label()
        self.transform = transform
        self.to_tensor = T.Compose([
            T.ToTensor()
        ])

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

     




