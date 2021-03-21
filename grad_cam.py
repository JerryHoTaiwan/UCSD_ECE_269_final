#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os

class _BaseWrapper(object):
    """
    Please modify forward() and backward() depending on your task.
    """

    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, idx):
        one_hot = torch.zeros((1, self.logits.size()[-1])).float()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)[0]
        return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)

    def backward(self, idx):
        """
        Class-specific backpropagation

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. self.logits[:, idx].backward(retain_graph=True)
        3. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = self._encode_one_hot(idx)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super().forward(self.image)

    def generate(self):
        gradient = self.image.grad.cpu().clone().numpy()
        self.image.grad.zero_()
        return gradient.transpose(0, 2, 3, 1)[0]


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=[]):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers

        def forward_hook(module, input, output):
            # Save featuremaps
            self.fmap_pool[id(module)] = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # Save the gradients correspond to the featuremaps
            self.grad_pool[id(module)] = grad_out[0].detach()

        # If any candidates are not specified, the hook is registered to all the layers.
        for module in self.model.named_modules():
            if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
                self.handlers.append(module[1].register_forward_hook(forward_hook))
                self.handlers.append(module[1].register_backward_hook(backward_hook))

    def _find(self, pool, target_layer):
        for key, value in pool.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError("Invalid layer name: {}".format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer, manual_str=""):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)


        gcam = (fmaps[0] * weights[0]).sum(dim=0)

        if target_layer == 'layer3':
            np_gcam = gcam.cpu().numpy()
            gmin = np.percentile(np_gcam, 50)
        else:
            gmin = 0.0
            np_gcam = gcam.cpu().numpy()
            gmin = np.percentile(np_gcam, 50)

        gcam = torch.clamp(gcam, min=gmin)
        gcam -= gcam.min()
        gcam /= gcam.max()

        response_list = list()
        response_list_ori = list()
        for j in range(fmaps.shape[1]):
            res_map = fmaps[0, j, :, :].cpu().numpy().reshape(fmaps.shape[2], fmaps.shape[3], 1)
            res_map -= np.amin(res_map)
            res_map /= np.amax(res_map)
            color_res_map = cv2.applyColorMap(np.uint8(res_map * 255.0), cv2.COLORMAP_JET)
            response_list_ori.append(color_res_map)
            color_res_map = cv2.resize(color_res_map, (576, 240))
            response_list.append(color_res_map)
            # cv2.imwrite('./vis_tmi/{}/{}/response_{}.png'.format(imgname, target_layer, j), color_res_map)
        
        h, w, _c = response_list[0].shape

        pad = 45
        grid_w = (w+pad) * 8
        grid_h = (h+pad) * len(response_list) // 8
        # fig = plt.figure(figsize=(grid_h//h, grid_w//w))
        # fig, ax = plt.subplots(grid_h // h, grid_w // w, figsize=(h * 2, w * 2))

        grid_array = np.zeros((grid_h, grid_w, 3)) + 255
        
        # manual_idx = [0, 2, 7, 8, 11, 13, 14, 20, 24, 28] # 20160714(#277)_P1__1
        if len(manual_str) > 0:
            manual_idx = [int(i) for i in manual_str.split(",")]
        else:
            manual_idx = list()
        manual_list = []

        for j, res in enumerate(response_list):
            x = j % 8
            y = j // 8
            grid_array[y*(h+pad):y*(h+pad)+h, x*(w+pad):x*(w+pad)+w, :] = res
            cv2.putText(grid_array, str(j), (x*(w+pad)+w//2, y*(h+pad)+h+pad//2+15), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 1, cv2.LINE_AA)
            if len(manual_idx) > 0 and j in manual_idx:
                manual_list.append(res)

            """
            if target_layer == 'layer1' or target_layer == 'layer2':
                cv2.imwrite("vis_cmp2/{}/{}/{}/res_{}.png".format(imgname, fold, target_layer, str(j)), res)
            """

            # bgr_res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            
            # if j < 2:
                # fig.add_subplot(grid_h // h, grid_w // w, j + 1)
                # plt.imshow(bgr_res)
                # ax[y][x].imshow(bgr_res)

        # plt.savefig('./vis_tmi/{}/{}/response_grid_plt.png'.format(imgname, target_layer))
        # plt.close()
        # return gcam.cpu().numpy()

        if len(manual_idx) > 0:
            avg_map = np.array(manual_list)
            print (avg_map.shape)
            avg_map = np.mean(avg_map, axis=0)
            avg_map -= np.amin(avg_map)
            avg_map /= np.amax(avg_map)
            avg_map *= 255
            avg_map = avg_map.astype(np.uint8)
            avg_map = cv2.cvtColor(avg_map, cv2.COLOR_RGB2GRAY)
           #cv2.imwrite("vis_tmi2/{}/{}/response_avg_{}.png".format(imgname, fold, target_layer), avg_map)

        # cv2.imwrite("vis_cmp3/{}/{}/grid_{}.png".format(imgname, fold, target_layer), grid_array)

        return gcam.cpu().numpy()
