import torch
import albumentations as A
from .pascal import VOCSegDataset

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


def build_loader(voc_dir, train_image_size=[256, 256], batch_size=16, num_workers=1):
    

    train_transform = build_transform(size=train_image_size, is_train=True)
    val_transform = build_transform(size=train_image_size, is_train=False)

    dataset_train, dataset_val = build_dataset(voc_dir, train_transfomr=train_transform, val_transform=val_transform)

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



def build_dataset(voc_dir, train_transfomr=None, val_transform=None):
    
    dataset_train = VOCSegDataset(voc_dir, is_train=True, transform=train_transfomr)
    dataset_val = VOCSegDataset(voc_dir, is_train=False, transform=val_transform)

    return dataset_train, dataset_val

def build_transform(size=[256, 256], is_train=True):

    if is_train:
        transform = A.Compose([
            A.RandomCrop(width=size[0], height=size[1]),
            A.HorizontalFlip(),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        ])
    else:
        transform = A.Compose([
            A.RandomCrop(width=size[0], height=size[1]),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        ])
    
    return transform