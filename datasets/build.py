import torch
import albumentations as A
from .pascal import VOCSegDataset

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

def build_loader(config):

    train_transform = build_transform(config, is_train=True)
    val_transform = build_transform(config, is_train=False)

    dataset_train, dataset_val = build_dataset(
        config.DATASET.root_dataset, 
        config.DATASET.image_size, 
        train_transfomr=train_transform, 
        val_transform=val_transform
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATASET.batch_size,
        num_workers=config.TRAIN.workers,
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATASET.batch_size,
        num_workers=config.TRAIN.workers,
        shuffle=False
    )

    return data_loader_train, data_loader_val

def build_dataset(voc_dir, image_size, train_transfomr=None, val_transform=None):
    
    dataset_train = VOCSegDataset(
        voc_dir, 
        image_size, 
        is_train=True, 
        transform=train_transfomr
    )
    dataset_val = VOCSegDataset(
        voc_dir, 
        image_size, 
        is_train=False, 
        transform=val_transform
    )

    return dataset_train, dataset_val

def build_transform(config, is_train=True):

    width = config.DATASET.image_size[0]
    height = config.DATASET.image_size[1]

    if is_train:
        transform = A.Compose([
            A.RandomCrop(width=width, height=height),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        ])
    else:
        transform = A.Compose([
            A.RandomCrop(width=width, height=height),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        ])
    
    return transform