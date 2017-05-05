__author__ = 'SherlockLiao'

import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
    '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir)
               if os.path.isdir(os.path.join(dir, d))]

    classes.sort(key=lambda x: int(x))  # file name is number
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


def resize(raw_img, target_size=(299, 299)):
    fx = target_size[0] / raw_img.shape[0]
    fy = target_size[1] / raw_img.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(raw_img, None, fx=fx, fy=fy,
                     interpolation=cv2.INTER_CUBIC)
    out_img = np.ones((target_size[0], target_size[1], 3)) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = (target_size[1] - w) / 2
    y = (target_size[0] - h) / 2
    x = int(x)
    y = int(y)
    out_img[y:y+h, x:x+w] = img
    return out_img


class Folder(Dataset):
    def __init__(self, root, target_size=(299, 299),
                 transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root
                               + "\n" "Supported image extensions are: "
                               + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.target_size = target_size
        self.classes = classes
        self.class_ti_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if target_size is not None:
            # img = resize(img, self.target_size)
            img = img.resize(self.target_size)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
