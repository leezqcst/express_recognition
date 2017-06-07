__author__ = 'SherlockLiao'

import time
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='saved model name')
parser.add_argument('--epoch', type=int, default=7000,
                    help='training epoches')
opt = parser.parse_args()
print(opt)

root = '/home/node/Documents/express_recognition/data/'
img_root = root + 'train/telephone_4'
txt_root = root + 'train/telephone_label_train_4.txt'

dset = dataset.Dataset(img_root=img_root, txt_root=txt_root)

dataloader = DataLoader(dset, batch_size=64, shuffle=True, num_workers=6,
                        collate_fn=dataset.alignCollate())


nh = 100
alphabet = '0123456789'
nclass = len(alphabet) + 1
nc = 1

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()

mynet = crnn.CRNN(32, nc, nclass, nh)

mynet.cuda()

loss_avg = utils.averager()
optimizer = optim.Adadelta(mynet.parameters(), lr=1e-3)


def trainBatch(crnn, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    image = Variable(cpu_images).cuda()
    t, le = converter.encode(cpu_texts)
    text = Variable(t)
    length = Variable(le)
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    # preds on cuda, text, preds_size, length is not on cuda
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


since = time.time()
for epoch in range(opt.epoch):
    train_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        for p in mynet.parameters():
            p.requires_grad = True
        mynet.train()

        cost = trainBatch(mynet, criterion, optimizer)
        loss_avg.add(cost)
        i += 1
        if (epoch+1) % 1 == 0:
            if i % 30 == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch+1, opt.epoch, i, len(dataloader), loss_avg.val()))
                loss_avg.reset()
    if (epoch+1) % 1 == 0:
        time_elipse = time.time() - since
        print('Time: {:.0f}s'.format(time_elipse))
        since = time.time()
#         if i % opt.valInterval == 0:
#             val(crnn, test_dataset, criterion)

        # do checkpointing
#         if i % opt.saveInterval == 0:
#             torch.save(
#                 crnn.state_dict(),
#                 '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
print('Finish Training!')
print()
save_path = root + 'model_save/telephone/' + opt.model + '.pth'
torch.save(mynet.state_dict(), save_path)
