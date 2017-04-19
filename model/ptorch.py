import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import time


img_transform = {
    'train': transforms.Compose([
            transforms.Scale(150),
            transforms.CenterCrop(299),
            transforms.ToTensor()
        ]),
    'val': transforms.Compose([
            transforms.Scale(150),
            transforms.CenterCrop(299),
            transforms.ToTensor()
        ])
}

root_path = '../data'

batch_size = 26
#读取数据文件夹
dset = {
    'train': ImageFolder(os.path.join(root_path, 'train/province'), transform=img_transform['train']),
    'val': ImageFolder(os.path.join(root_path, 'val/province'), transform=img_transform['val'])
}

#读取数据
dataloader = {
    'train': DataLoader(dset['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(dset['val'], batch_size=batch_size, num_workers=4)
}

data_size = {
    x: len(dataloader[x].dataset.imgs)
    for x in ['train', 'val']
}

img_classes = dataloader['train'].dataset.classes

use_gpu = torch.cuda.is_available()

# mynet = vgg16(3, 6)
mynet = torchvision.models.inception_v3()

mynet.fc = nn.Linear(2048, 30)

if use_gpu:
    mynet = mynet.cuda()

# define optimizer and loss
optimizer = optim.SGD(mynet.parameters(), lr=1e-3, momentum=0.9)
# 随机梯度下降，之后可以选择别的速度更快的如rmsprop
criterion = nn.CrossEntropyLoss()

num_epoch = 20

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
        out, _ = mynet(img)
        loss = criterion(out, label)
        _, pred = torch.max(out, 1)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * label.size(0)
        num_correct = torch.sum(pred==label)
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
