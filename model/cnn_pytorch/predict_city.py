import os
import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from inceptionresnet import InceptionResnetV2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True, help='image name')
opt = parser.parse_args()
# print(opt)
use_gpu = torch.cuda.is_available()
model_path = '/home/sherlock/Desktop/inception-resnet.pth'

print('Start loading model...')
citymodel = InceptionResnetV2(64)
citymodel.load_state_dict(torch.load(model_path))
citymodel.eval()
if use_gpu:
    citymodel = citymodel.cuda()
print('model loading successful!')
print()

print('Start loading images...')
root = '/home/sherlock/Desktop'
img_path = os.path.join(root, opt.img)
img = Image.open(img_path).convert('RGB')
img = img.resize((299, 299))
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)
if use_gpu:
    img = Variable(img, volatile=True).cuda()
else:
    img = Variable(img)
print('image loading successful!')
print()

print('Start predicting!')
preds = citymodel(img)
preds = F.softmax(preds)
prob, label = torch.topk(preds, 3, 1)
prob = prob.data[0]
label = label.data[0]
print('top 3 probability: {}'.format(prob))
print('top 3 label: {}'.format(label))
print('Finish predicting')
