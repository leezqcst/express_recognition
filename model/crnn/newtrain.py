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
import utils
import dataset
import crnn

root = '/home/sherlock/Documents/express_recognition/data/'
img_root = root + 'train/telephone_1'
txt_root = root + 'train/telephone_label_train_1.txt'

dset = dataset.Dataset(img_root=img_root, txt_root=txt_root)

dataloader = DataLoader(dset, batch_size=32, shuffle=True, num_workers=4,
                        collate_fn=dataset.alignCollate())


nh = 100
alphabet = '0123456789'
nclass = len(alphabet)
nc = 1
epoches = 10000

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()

mynet = crnn.CRNN(32, nc, nclass, nh)
mynet.cuda()

loss_avg = utils.averager()
optimizer = optim.Adadelta(mynet.parameters(), lr=1e-3)

for epoch in range(epoches):
    for i, data in enumerate(dataloader):
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        image = Variable(cpu_images).cuda()
        t, le = converter.encode(cpu_texts)
        text = Variable(t)
        length = Variable(le)
        preds = mynet(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        mynet.zero_grad()
        cost.backward()
        optimizer.step()
        print('{}/{}, Loss: {}'.format(epoch, epoches, cost.data[0]))

print('Finish Training!')
print()
save_path = root + 'model_save/telephone/' + opt.model + '.pth'
torch.save(mynet.state_dict(), save_path)
