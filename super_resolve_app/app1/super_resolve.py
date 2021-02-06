from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

from django.conf import settings
from .model import Net


def super_resolve(input_url, output_url):
    # Training settings

    input_image = input_url

    img = Image.open(input_image).convert('YCbCr')
    y, cb, cr = img.split()

    model_path = settings.BASE_DIR + '/app1/model_epoch_30.pth'

    the_model = Net(3)
    the_model.load_state_dict(torch.load(model_path))

    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    out = the_model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    out_img.save(output_url)
    print('output image saved to ', output_url)
