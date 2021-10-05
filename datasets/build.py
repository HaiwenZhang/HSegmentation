import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .pascal import VOCSegDataset

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


def build_loader(voc_dir, train_image_size=[256, 256], batch_size=16, num_workers=1):
    

    #train_transform = build_transform(size=train_image_size, is_train=True)


    train_transform = A.Compose([
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2(),
    ])


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
            ),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.RandomCrop(width=size[0], height=size[1]),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
            ToTensorV2(),
        ])
    
    return transform