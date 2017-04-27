__author__ = 'SherlockLiao'

import os
from tqdm import tqdm
import h5py
import numpy as np
import model as ml
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

parse = argparse.ArgumentParser()
parse.add_argument('--model', required=True,
                   help='model to extract feature, vgg, \
                   resnet152, inceptionv3')
parse.add_argument('--batch_size', type=int, default=32)
parse.add_argument('--num_workers', type=int, default=4)
parse.add_argument('--phase', required=True, help='train or val')
opt = parse.parse_args()
print(opt)

img_transform = transforms.Compose([
    transforms.Scale(150),
    transforms.CenterCrop(299),
    transforms.ToTensor()
])

root_path = '/home/sherlock/Documents/express_recognition/data'
dset = {
    'train': ImageFolder(os.path.join(root_path, 'train/province'),
                         transform=img_transform),

    'val': ImageFolder(os.path.join(root_path, 'validation/province'),
                       transform=img_transform)
}

batch_size = opt.batch_size
num_workers = opt.num_workers

dataloader = {
    'train': DataLoader(dset['train'], batch_size=batch_size, shuffle=False,
                        num_workers=num_workers),
    'val': DataLoader(dset['val'], batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)
}


def createDataset(outputPath, model, phase):
    """
    Create h5py dataset for feature extraction.

    ARGS:
        outputPath    : h5py output path
        model         : used model
        labelList     : list of corresponding groundtruth texts
    """
    feature_net = ml.feature_extraction(model)
    feature_net.cuda()
    feature_map = torch.FloatTensor()
    label_map = torch.LongTensor()
    for data in tqdm(dataloader[phase]):
        img, label = data
        img = Variable(img, volatile=True).cuda()
        out = feature_net(img)
        feature_map = torch.cat((feature_map, out.cpu().data), 0)
        label_map = torch.cat((label_map, label), 0)
    feature_map = feature_map.numpy()
    label_map = label_map.numpy()
    file_name = '_feature_{}.hd5f'.format(model)
    h5_path = os.path.join(outputPath, phase) + file_name
    with h5py.File(h5_path, 'w') as h:
        h.create_dataset('data', data=feature_map)
        h.create_dataset('label', data=label_map)


class h5Dataset(Dataset):

    def __init__(self, h5py_list):
        label_file = h5py.File(h5py_list[0], 'r')
        self.label = torch.from_numpy(label_file['label'].value)
        self.nSamples = self.label.size(0)
        temp_dataset = torch.FloatTensor()
        for file in h5py_list:
            h5_file = h5py.File(file, 'r')
            dataset = torch.from_numpy(h5_file['data'].value)
            temp_dataset = torch.cat((temp_dataset, dataset), 1)

        self.dataset = temp_dataset

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        data = self.dataset[index]
        label = self.label[index]
        return (data, label)
