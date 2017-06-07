__author__ = 'SherlockLiao'

import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import utils
import dataset
import crnn

root = '/home/sherlock/Documents/express_recognition/data/'
img_root = root + 'train/telephone_1'
txt_root = root + 'train/telephone_label_train_1.txt'

dset = dataset.Dataset(img_root=img_root, txt_root=txt_root)

dataloader = DataLoader(dset, batch_size=64, shuffle=True, num_workers=4,
                        collate_fn=dataset.alignCollate())


alphabet = '0123456789'
nclass = len(alphabet)
nc = 1
epoches = 100

converter = utils.strLabelConverter(alphabet)
criterion = nn.NLLLoss()

cnn = crnn.ExtractCNN(32, nc)
cnn.cuda()
encoder = crnn.Encoder(512, 100)
encoder.cuda()
decoder = crnn.Decoder(100, nclass)
decoder.cuda()

parm = list(cnn.parameters()) + list(encoder.parameters()) \
     + list(decoder.parameters())
optimizer = optim.Adadelta(parm, lr=1e-3)

for epoch in range(epoches):
    for i, data in enumerate(dataloader):
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        image = Variable(cpu_images).cuda()
        t, le = converter.encode(cpu_texts)
        t = t.view(batch_size, -1)
        label = Variable(t).cuda()
        loss = 0
        # CNN feature vectors
        cnn_out = cnn(image)
        # Encoder
        hidden = encoder.initHidden(batch_size)
        for i in range(cnn_out.size()[0]):
            en_input = cnn_out[i, :, :]
            en_input = en_input.unsqueeze(0)
            en_out, hidden_out = encoder(en_input, hidden)
            hidden = hidden_out
        # Decoder
        de_hidden = hidden
        de_input = hidden
        for le in range(11):
            prob, de_hidden_out = decoder(de_input, de_hidden)
            de_input = de_hidden_out
            de_hidden = de_hidden_out
            temp_label = label[:, le]
            loss += criterion(prob, temp_label)
        loss.backward()
        optimizer.step()
    print('{}/{}, Loss: {}'.format(epoch, epoches, loss.data[0]))

print('Finish Training!')
print()
cnn_path = root + 'model_save/telephone/excnn.pth'
encoder_path = root + 'model_save/telephone/encoder.pth'
decoder_path = root + 'model_save/telephone/decoder.pth'
torch.save(cnn.state_dict(), cnn_path)
torch.save(encoder.state_dict(), encoder_path)
torch.save(decoder.state_dict(), decoder_path)
