import torch

from models import build_model


if __name__ == "__main__":
    num_class = 21
    dilate_scale = 16

    model = build_model(num_class, dilate_scale=dilate_scale)
    img = torch.rand((2, 3, 256, 256))
    o = model(img)
    print(o.shape)
    # print(model)