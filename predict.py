from torch.utils.data import dataset
from tqdm import tqdm
import os
import argparse

from models import build_predict_model

from torch.utils import data

from torchvision import transforms as T


import torch
import torch.nn as nn

from PIL import Image
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--ckpt", type=str, help='model checkpoint')
    parser.add_argument("--output", type=str, help="Predict image directory")

    return parser



def read_voc_val_data_info(voc_dir):
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'val.txt')
    with open(txt_fname, 'r') as f:
        img_lists = f.read().split()
    return img_lists

def read_image(path):
    img = Image.open(path).convert('RGB')
    return img


def main():
    opts = get_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup data
    images = read_voc_val_data_info(opts.input)
    
    # Set up model

    model = build_predict_model(opts.ckpt)
    model.to(device)

    transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),

        ])

    if opts.output is not None:
        os.makedirs(opts.output, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for fname in tqdm(images):
            path = os.path.join(opts.input, 'JPEGImages',  f'{fname}.jpg')
            image = read_image(path)
            image = transform(image).unsqueeze(0).to(device)
    
            pred = model(image).max(1)[1].cpu().numpy()[0] # HW

            print(pred.shape)

if __name__ == '__main__':
    main()