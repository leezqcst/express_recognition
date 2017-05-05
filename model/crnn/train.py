
# coding: utf-8

# In[1]:

import time
import random
import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import crnn


# In[2]:
root = '/home/sherlock/'
img_root = root + 'Documents/express_recognition/data/train/telephone'
txt_root = root + 'Documents/express_recognition/data/train/telephone_label_train.txt'

dset = dataset.Dataset(img_root=img_root, txt_root=txt_root)

dataloader = DataLoader(dset, batch_size=32, num_workers=4,
                        collate_fn=dataset.alignCollate())


nh = 100
alphabet = '0123456789'
nclass = len(alphabet) + 1
nc = 1
epoches = 10000

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()

mynet = crnn.CRNN(32, nc, nclass, nh)

mynet.cuda()

criterion = criterion.cuda()

loss_avg = utils.averager()
optimizer = optim.Adadelta(mynet.parameters(), lr=1e-3)


def trainBatch(crnn, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    image = torch.FloatTensor(cpu_images)
    image = Variable(image).cuda()
    t, le = converter.encode(cpu_texts)
    text = torch.IntTensor(t)
    text = Variable(text)
    length = torch.IntTensor(le)
    length = Variable(length)
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(epoches):
    train_iter = iter(dataloader)
    i = 0
    since = time.time()
    while i < len(dataloader):
        for p in mynet.parameters():
            p.requires_grad = True
        mynet.train()

        cost = trainBatch(mynet, criterion, optimizer)
        loss_avg.add(cost)
        i += 1
        if (epoch+1) % 500 == 0:
            if i % 10 == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch+1, epoches, i, len(dataloader), loss_avg.val()))
                loss_avg.reset()
#     time_elipse = time.time() - since
#     print('Time: {:.0f}'.format(time_elipse))
#         if i % opt.valInterval == 0:
#             val(crnn, test_dataset, criterion)

        # do checkpointing
#         if i % opt.saveInterval == 0:
#             torch.save(
#                 crnn.state_dict(),
#                 '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
save_path = './crnn_model.pth'
torch.save(mynet.state_dict(), save_path)
