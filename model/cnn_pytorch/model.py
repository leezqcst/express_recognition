__author__ = 'SherlockLiao'

import torch
import torchvision.models as models
from torch import nn
from torch.autograd import Variable


def inception_net(n_classes):
    incep_model = models.inception_v3()
    fc_layer = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(True),
        nn.Linear(1000, 500),
        nn.ReLU(True),
        nn.Linear(500, n_classes)
    )
    incep_model.fc = fc_layer
    return incep_model


def vgg_net(n_classes):
    vgg_model = models.vgg19()
    fc_layer = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 1000),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(1000, 500),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(500, n_classes)
    )
    vgg_model.classifier = fc_layer
    return vgg_model


def resnet_net(n_classes):
    resnet_model = models.resnet152()
    fc_layer = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(True),
        nn.Linear(1000, 500),
        nn.ReLU(True),
        nn.Linear(500, n_classes)
    )
    resnet_model.fc = fc_layer
    return resnet_model


class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
