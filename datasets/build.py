import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .pascal import VOCSegDataset

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


def build_loader(voc_dir, train_image_size, batch_size=16, num_workers=1):
    

    train_transforms = build_transform(size=train_image_size, is_train=True)
    val_transforms = build_transform(is_train=False)

    dataset_train, dataset_val = build_dataset()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )

    return data_loader_train, data_loader_val



def build_dataset(voc_dir, train_transfomrs=None, val_transforms=None):
    
    dataset_train = VOCSegDataset(voc_dir, is_train=True, transforms=train_transfomrs)
    dataset_val = VOCSegDataset(voc_dir, is_train=False, transfromrs=val_transforms)

    return dataset_train, dataset_val


def build_transform(size=[256, 256], is_train=True):

    if is_train:
        transforms = A.Compose([
            A.RandomCrop(width=size[0], height=size[1]),
            A.HorizontalFlip(),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
            ToTensorV2()
        ])
    else:
        transforms = A.Compose([
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
            ToTensorV2()
        ])
    
    return transforms