__authoer__ = 'SherlockLiao'

import torch
import torchvision.models as models
from torch import nn


class feature_extraction(nn.Module):
    def __init__(self, model):
        super(feature_extraction, self).__init__()
        if model == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(9))
        elif model == 'inceptionv3':
            inceptionv3 = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inceptionv3.children())[:-1])
            self.feature._modules.pop('13')
            self.feature.add_module('global average', nn.AvgPool2d(35))
        elif model == 'resnet152':
            resnet152 = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet152.children())[:-1])

    def forward(self, input):
        '''
        model includes vgg, inceptionv3, resnet152
        '''
        self.feature.cuda()
        out = self.feature(input)
        out = out.view(out.size(0), -1)
        return out


class fcnet(nn.Module):
    def __init__(self, dim, n_classes):
        super(fcnet, self).__init__()
        classifier = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(true),
            nn.Linear(1000, n_classes)
        )
        self.fc = classifier

    def forward(self, input):
        out = self.fc(input)
        return out
