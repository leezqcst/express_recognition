import os
import argparse

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import model

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='model choice: inceptionV3, \
                    inceptionV4, resnet101, resnet152')
parser.add_argument('--path', required=True,
                    help='data set path: province, city')
parser.add_argument('--n_classes', type=int, help='numbers of classes',
                    default=30)
parser.add_argument('--num_worker', type=int,
                    help='number of data loading workers', default=4)
opt = parser.parse_args()
print(opt)

if opt.model == 'inceptionV3':
    img_size = 299
if opt.model == 'inceptionV4':
    img_size = 299
if opt.model == 'resnet':
    img_size = 299

img_transform = transforms.Compose([
    transforms.Scale(150),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])

root_path = '/home/sherlock/Documents/express_recognition/data'
model_path = os.path.join(root_path,
                          'model_save/' + opt.path + '/' + opt.model + '.pth')

batch_size = 32
num_worker = opt.num_worker

dset = ImageFolder(os.path.join(root_path, 'validation/' + opt.path),
                   transform=img_transform)

dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                        num_workers=num_worker)

use_gpu = torch.cuda.is_available()

if opt.model == 'inceptionV3':
    mynet = model.inceptionV3(opt.n_classes)
if opt.model == 'inceptionV4':
    mynet = model.InceptionV4(opt.n_classes)
if opt.model == 'resnet101':
    mynet = model.resnet(opt.n_classes, 101)
if opt.model == 'resnet152':
    mynet = model.resnet(opt.n_classes, 152)

mynet.load_state_dict(torch.load(model_path))

if use_gpu:
    mynet = mynet.cuda()

mynet.eval()
num_correct = 0.0
total = 0.0
for data in dataloader:
    img, label = data
    if use_gpu:
        img = Variable(img, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)

    out = mynet(img)
    _, pred = torch.max(out.data, 1)
    num_correct += (pred.cpu() == label).sum()
    total += label.size(0)
print('Acc: {:.6f}'.format(num_correct / total))
