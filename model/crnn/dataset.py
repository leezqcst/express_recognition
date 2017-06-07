"define dataset"

__author__ = 'SherlockLiao'

import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('L')


def image_label(img_root, txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    image = [os.path.join(img_root, i.split()[0]) for i in lines]
    label = [i.split()[1] for i in lines]
    return image, label


class Dataset(Dataset):
    def __init__(self, img_root, txt_root, target_size=None,
                 transform=None, loader=default_loader):
        self.image, self.label = image_label(img_root, txt_root)
        self.img_root = img_root
        self.txt_root = txt_root
        self.target_size = target_size
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.image[index]
        target = self.label[index]
        img = self.loader(img_path)
        if self.target_size is not None:
            img = img.resize(self.target_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.mul_(-1).add_(1)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self, imgH=32, imgW=256, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels
