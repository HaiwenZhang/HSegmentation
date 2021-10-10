import torch
import cv2
import albumentations as A
from .pascal import VOCSegDataset

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

def build_loader(config):

    train_transform = build_transform(config, is_train=True)
    val_transform = build_transform(config, is_train=False)

    dataset_train, dataset_val = build_dataset(
        config.DATASET.root_dataset, 
        train_transfomr=train_transform, 
        val_transform=val_transform
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATASET.batch_size,
        num_workers=config.TRAIN.workers,
        shuffle=True,
        pin_memory=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATASET.batch_size,
        num_workers=config.TRAIN.workers,
        shuffle=False,
        pin_memory=True
    )

    return data_loader_train, data_loader_val

def build_dataset(voc_dir, train_transfomr=None, val_transform=None):
    
    dataset_train = VOCSegDataset(
        voc_dir, 
        is_train=True, 
        transform=train_transfomr
    )
    dataset_val = VOCSegDataset(
        voc_dir, 
        is_train=False, 
        transform=val_transform
    )

    return dataset_train, dataset_val

def build_transform(config, is_train=True):

    resize_height = config.DATASET.image_size[1]
    resize_width = config.DATASET.image_size[0]

    crop_height = config.DATASET.crop_size[0]
    crop_width = config.DATASET.crop_size[1]

    if is_train:
        transform = A.Compose([
            A.Resize(height=resize_height, width=resize_width),
            A.CenterCrop(height=crop_height, width=crop_width),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        ])
    else:
        transform = A.Compose([
            A.Resize(height=resize_height, width=resize_width),
            A.CenterCrop(height=crop_height, width=crop_width),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        ])
    
    return transform