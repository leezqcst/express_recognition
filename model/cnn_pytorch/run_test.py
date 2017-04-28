import os
import argparse

import torch
from torch import nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import model
from inceptionresnet import InceptionResnetV2

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='model choice: inceptionV3, \
                    inceptionV4, resnet101, resnet152, inception-resnet')
parser.add_argument('--path', required=True,
                    help='data set path: province, city')
parser.add_argument('--n_classes', type=int, help='numbers of classes',
                    default=30)
parser.add_argument('--num_worker', type=int,
                    help='number of data loading workers', default=4)
opt = parser.parse_args()
print(opt)

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
if opt.model == 'inception-resnet':
    mynet = InceptionResnetV2(opt.n_classes)

mynet.load_state_dict(torch.load(model_path))
criterion = nn.CrossEntropyLoss()

if use_gpu:
    mynet = mynet.cuda()

mynet.eval()
num_correct = 0.0
total = 0.0
eval_loss = 0.0
for data in dataloader:
    img, label = data
    img = Variable(img, volatile=True).cuda()
    label = Variable(label, volatile=True).cuda()
    out = mynet(img)
    _, pred = torch.max(out.data, 1)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    num_correct += (pred.cpu() == label.data.cpu()).sum()
    total += label.size(0)
print('Loss: {:.6f} Acc: {:.6f}'.format(eval_loss / total,
                                        num_correct / total))
