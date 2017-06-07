import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ExtractCNN(nn.Module):
    def __init__(self, imgH, nc, leakyRelu=False):
        super(ExtractCNN, self).__init__()

        assert imgH % 16 == 0, 'image height must be a multiple of 16'

        kernel_size = [3, 3, 3, 3, 3, 3, 2]
        padding_size = [1, 1, 1, 1, 1, 1, 0]
        stride_size = [1, 1, 1, 1, 1, 1, 1]
        n_filter = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else n_filter[i - 1]
            nOut = n_filter[i]
            cnn.add_module('conv{}'.format(i),
                           nn.Conv2d(nIn,
                                     nOut,
                                     kernel_size[i],
                                     stride_size[i],
                                     padding_size[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{}'.format(i),
                               nn.BatchNorm2d(nOut))

            if leakyRelu:
                cnn.add_module('leakyrelu{}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))

            else:
                cnn.add_module('relu{}'.format(i),
                               nn.ReLU(inplace=True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x128
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1),
                       nn.MaxPool2d((2, 2)))  # 128x16x64
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2)))

        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2),
                                    (2, 1),
                                    (0, 1)))  # 512x1x34
        convRelu(6, True)  # 512x2x16
        # cnn.add_module('avgpooling{0}'.format(4),
        #                nn.AvgPool2d((2, 1),
        #                             (1, 1),
        #                             (0, 0)))  # 512x1x16

        self.cnn = cnn

    def forward(self, input):
        # conv features map
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [length, batch_size, channel]
        return conv


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=n_layers)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size,
                                      self.hidden_size))
        return hidden.cuda()


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_layers=n_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        assert input.size()[0] == 1, 'length must be 1'
        input = F.relu(input)
        output, hidden = self.gru(input, hidden)
        output = output.squeeze(0)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size,
                                      self.hidden_size))
        return hidden.cuda()
