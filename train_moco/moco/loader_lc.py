# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang
from PIL import ImageFilter, Image
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import time
import torch
import numpy as np
import random
import math

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.resize = RandomResizedCrop_Self(224, scale=(0.2, 1.))

    def __call__(self, x):
        q, k = self.resize(x, x.copy())
        q = self.base_transform(q)    
        k = self.base_transform(k)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class RandomResizedCrop_Self(torch.nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), delta=0.2, same_size=False):
        super(RandomResizedCrop_Self, self).__init__()
        self.t = transforms.RandomResizedCrop(size, scale)
        self.delta = delta
        self.same_size = same_size
        print("Delta:", self.delta)

    def forward(self, x, x_copy): # larger, smaller
        width, height = x.size
        area = height * width
        min_area = area * self.t.scale[0]


        # view 1
        i, j, h, w = self.t.get_params(x, self.t.scale, self.t.ratio)
        view1 = F.resized_crop(x, i, j, h, w, self.t.size, self.t.interpolation)


        # enlarged view 1
        margin_h = int(self.delta*h)
        margin_w = int(self.delta*w)
        l_top = max(i-margin_h, 0)
        l_bottom = min(i+h+margin_h, height)
        l_left = max(j-margin_w, 0)
        l_right = min(j+w+margin_w, width)

        l_h = l_bottom-l_top
        l_w = l_right-l_left
        l_area = l_h * l_w
        v2_scale = (min_area/l_area, 1.0)


        # view 2
        if self.same_size:
            h2 = h 
            w2 = w
        else:
            try:
                h2 = -1
                w2 = -1
                log_ratio = torch.log(torch.tensor(self.t.ratio))
                for _ in range(10):
                    target_area = l_area * torch.empty(1).uniform_(v2_scale[0], v2_scale[1]).item()
                    aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
                    h2 = int(round(math.sqrt(target_area / aspect_ratio)))
                    w2 = int(round(math.sqrt(target_area * aspect_ratio)))
                    if 0 < w2 <= l_w and 0 < h2 <= l_h:
                        break
                    else:
                        h2 = -1
                        w2 = -1
                # fallback
                if h2==-1 or w2==-1:
                    h2 = h 
                    w2 = w
            except:
                raise
        # i2 j2
        try:
            i2 = np.random.randint(l_top, l_bottom-h2+1)
        except:
            raise
        try:
            j2 = np.random.randint(l_left, l_right-w2+1)
        except:
            raise
        view2 = F.resized_crop(x_copy, i2, j2, h2, w2, self.t.size, self.t.interpolation)


        # q: view 1
        if random.random()<0.5:
            q = view1
            k = view2
        else: # k: view 1
            q = view2 
            k = view1
        return q, k