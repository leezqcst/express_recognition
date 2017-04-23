import os

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import model

img_transform = transforms.Compose([
    transforms.Scale(150),
    transforms.CenterCrop(299),
    transforms.ToTensor()
])

root_path = '/home/sherlock/Documents/express_recognition/data'
model_path = os.path.join(root_path, 'model_save/example.pth')

batch_size = 32
num_worker = 4

dset = ImageFolder(os.path.join(root_path, 'val/province'),
                   transform=img_transform)

dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                        num_workers=num_worker)

use_gpu = torch.cuda.is_available()
mynet = model.inception_net(30)
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
