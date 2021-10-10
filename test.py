import torch
import os
import cv2
from tqdm import tqdm

from models import build_model
from datasets import VOCSegDataset, label2image
import albumentations as A


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

if __name__ == "__main__":

    voc_dir = "/home/haiwen/Datasets/VOC2012"

    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    ])


    dataset_train = VOCSegDataset(
        voc_dir,
        is_train=True, 
        transform=transform
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )
    with tqdm(total=len(data_loader_train)) as t:
        for idx, (data, target) in enumerate(data_loader_train):
            image = data[1,:].clone().detach().cpu().numpy()
            target = label2image(target, device="cpu")
            mask = target[1,:].clone().detach().cpu().numpy()

            valid_dir = os.path.join("output", "data")
            if not os.path.isdir(valid_dir):
                os.makedirs(valid_dir)

            image_path = os.path.join(valid_dir, f"idx_{idx}.jpg")
            mask_path = os.path.join(valid_dir, f"idx_{idx}_mask.jpg")
            
            image = image.transpose((1, 2, 0)) * 255
            cv2.imwrite(image_path, image)
            cv2.imwrite(mask_path, mask)
            
            t.set_description(f"Test Dataloder")
            t.update()