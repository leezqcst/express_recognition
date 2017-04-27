import os
import time
import argparse

import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import model

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='model choice: inceptionV3, inceptionV4, \
                    resnet101, resnet152')
parser.add_argument('--pretrained', type=bool,
                    help='if load pretrained weigths', default=False)
parser.add_argument('--path', required=True,
                    help='data set path: province, city')
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--epoch', type=int, required=True,
                    help='numbers of epoch')
parser.add_argument('--n_classes', type=int, help='numbers of classes',
                    default=30)
parser.add_argument('--num_worker', type=int,
                    help='number of data loading workers', default=4)
opt = parser.parse_args()
print(opt)

if opt.model == 'inceptionV3':
    img_size = 299
if opt.model == 'resnet101':
    img_size = 299
    n_layers = 101
if opt.model == 'resnet152':
    img_size = 299
    n_layers = 152
if opt.model == 'inceptionV4':
    img_size = 299

img_transform = {
    'train': transforms.Compose([
            transforms.Scale(150),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ]),
    'val': transforms.Compose([
            transforms.Scale(150),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
}

root_path = '/home/sherlock/Documents/express_recognition/data'

batch_size = opt.batch_size
num_epoch = opt.epoch
# 读取数据文件夹
dset = {
    'train': ImageFolder(os.path.join(root_path, 'train/' + opt.path),
                         transform=img_transform['train']),
    'val': ImageFolder(os.path.join(root_path, 'validation/' + opt.path),
                       transform=img_transform['val'])
}

# 读取数据
dataloader = {
    'train': DataLoader(dset['train'], batch_size=batch_size, shuffle=True,
                        num_workers=opt.num_worker),
    'val': DataLoader(dset['val'], batch_size=batch_size,
                      num_workers=opt.num_worker)
}

data_size = {
    x: len(dataloader[x].dataset.imgs)
    for x in ['train', 'val']
}

img_classes = dataloader['train'].dataset.classes

use_gpu = torch.cuda.is_available()

if opt.model == 'resnet101':
    mynet = model.resnet(opt.n_classes, n_layers=n_layers,
                         pretrained=opt.pretrained)
if opt.model == 'resnet152':
    mynet = model.resnet(opt.n_classes, n_layers=n_layers,
                         pretrained=opt.pretrained)
if opt.model == 'inceptionV3':
    mynet = model.inceptionV3(opt.n_classes)
if opt.model == 'inceptionV4':
    mynet = model.InceptionV4(opt.n_classes)

if use_gpu:
    mynet = mynet.cuda()

# define optimizer and loss
optimizer = optim.SGD(mynet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()


for epoch in range(num_epoch):
    print(epoch + 1)
    print('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    since = time.time()
    for i, data in enumerate(dataloader['train'], 1):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        # forward
        if opt.model == ('inceptionV3' or 'inceptionV4'):
            out, _ = mynet(img)
        else:
            out = mynet(img)
        loss = criterion(out, label)
        _, pred = torch.max(out, 1)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * label.size(0)
        num_correct = torch.sum(pred == label)
        running_acc += num_correct.data[0]
        if i % 50 == 0:
            print('Loss:{:.4f}, Acc: {:.4f}'.format(
                                            running_loss / (i * batch_size),
                                            running_acc / (i * batch_size)))
    running_loss /= data_size['train']
    running_acc /= data_size['train']
    elips_time = time.time() - since
    print('{}/{}, Loss:{:.4f}, Acc:{:.4f}, Time:{:.0f}s'.format(
                                                        epoch+1,
                                                        num_epoch,
                                                        running_loss,
                                                        running_acc,
                                                        elips_time))
    print()
print('Finish Training!')
print()
# validation
mynet.eval()
num_correct = 0.0
total = 0.0
for data in dataloader['val']:
    img, label = data
    img = Variable(img, volatile=True).cuda()

    out = mynet(img)
    _, pred = torch.max(out.data, 1)
    num_correct += (pred.cpu() == label).sum()
    total += label.size(0)
print('Acc:{}'.format(num_correct / total))
save_path = os.path.join(root_path,
                         'model_save/' + opt.path + '/' + opt.model + '.pth')
torch.save(mynet.state_dict(), save_path)
