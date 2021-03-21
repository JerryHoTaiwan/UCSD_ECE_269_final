import os
import sys
sys.path.insert(0, '../../')
import copy
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from sota.cnn.grad_cam import GradCAM
from sota.cnn.model_search_pcdarts import PCDARTSNetwork as Network
from sota.cnn.spaces import spaces_dict
import matplotlib.cm as cm
#from resnet import resnet18

parser = ArgumentParser()
parser.add_argument('--img_path', required=True)
parser.add_argument('--tens_path', required=True)
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--search_space', type=str, default='s3', help='searching space to choose from')
args = parser.parse_args()

def pre_process_image(x):
    return x.astype(np.float32)
    max_y, max_x = 200, 576
    if x.shape[0] >= max_y:
        xs = x[:max_y,:max_x]
    else:
        pad = np.zeros((max_y-x.shape[0], max_x))
        xs = x[:,:max_x]
        xs = np.concatenate((xs, pad), axis=0)

    return xs.astype(np.float32)


def save_gradcam(filename, gcam, raw_image):
    h, w = raw_image.shape
    raw_image[raw_image>400]=400
    raw_image -= np.amin(raw_image)
    raw_image /= np.amax(raw_image)
    raw_image *= 255
    # raw_image = (raw_image - np.amin(raw_image)) / (np.amax(raw_image) - np.amin(raw_image)) * 255
    stacked_raw_img = np.stack((raw_image, raw_image, raw_image), axis=-1)

    gcam_max = np.percentile(gcam, 98)
    gcam_min = np.percentile(gcam, 2)
    gcam = np.clip(gcam, gcam_min, gcam_max)
    gcam -= np.amin(gcam)
    gcam /= np.amax(gcam)

    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    # gcam = np.concatenate((gcam, stacked_raw_img), axis=0)
    gcam = gcam.astype(np.float) #+ stacked_raw_img.astype(np.float)/1.14
    # gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))

def main():
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, 10, args.layers, criterion, spaces_dict[args.search_space]).cuda()
    model.load_state_dict(torch.load("weights-3.pt"))
    model.eval()
    
    #for module in model.named_modules():
    #    print (module[0])

    # Open image
    raw_image = cv2.imread(args.img_path)

    tens = np.load(args.tens_path, allow_pickle=True)
    image = torch.from_numpy(tens).unsqueeze(0)#.unsqueeze(0)
    image = image.cuda()
    print (image.size())
    pred = model(image)
    print (pred)

    # GCAM
    gcam = GradCAM(model=model)
    predictions = gcam.forward(image)
    top_idx = predictions[0][1]
    print(predictions, len(predictions), top_idx)
    target_layer = "cells.19"
    gcam.backward(idx=top_idx)
    region = gcam.generate(target_layer=target_layer)
    cmap = cm.jet_r(region)[..., :3] * 255.0
    cmap = cv2.resize(cmap, (32, 32))
    blend = (cmap+raw_image)/2
    cv2.imwrite("blend_4.png", blend)
    print (region.shape, cmap.shape, raw_image.shape)

if __name__ == "__main__":
    main()



