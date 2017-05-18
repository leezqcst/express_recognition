import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.GRU(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()

        assert imgH % 16 == 0, 'image heigth must be a multiple of 16'

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
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features map
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [width, batchsize, channel]

        output = self.rnn(conv)

        return output
