# -*- coding: utf-8 -*-
"""
Author: Zongyue Qin
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import pickle

dft_content_features_layer = ['conv_10']
dft_style_features_layer = ['conv_1', 'conv_3', 'conv_5', 'conv_9']

def CenterCrop(x, crop_height, crop_width):
    _, height, width = x.shape
    start_h = height // 2 - crop_height // 2
    start_w = width // 2 - crop_width // 2
    return x[:,start_h:start_h+crop_height, start_w:start_w+crop_width]


class Gram(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        newx = x.view(x.size(0), x.size(1), -1)
        newx_T = torch.transpose(newx, 1, 2)
        return torch.matmul(newx, newx_T)

class model(nn.Module):
    
    def __init__(self, vgg = models.vgg19(pretrained=True), content_weight = 1,
                 style_weight = 1, content_features_layer = dft_content_features_layer,
                 style_features_layer = dft_style_features_layer):
        super().__init__()
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.content_w = content_weight
        self.style_w = style_weight
        self.content_f_layer = content_features_layer
        self.style_f_layer = style_features_layer
        self.content_criterion = nn.MSELoss()
        
    def forward(self, x, content, style):
        
        gram = Gram()
        content_feature = []
        style_feature = []
        content_feature_cnt = 0
        style_feature_cnt = 0
        i = 1
        for layer in self.vgg.features:
            x = layer(x)
            content = layer(content)
            style = layer(style)
            
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                
                if name in self.content_f_layer:
                    content_feature_cnt = content_feature_cnt + 1
                    content_feature.append([x.clone(), content.detach()])
                if name in self.style_f_layer:
                    style_feature_cnt = style_feature_cnt + 1
                    style_feature.append([gram(x.clone()), gram(style.detach()), 
                                          x.size(1), x.size(2) * x.size(3)])
                    
                if content_feature_cnt == len(self.content_f_layer) and\
                    style_feature_cnt == len(self.style_f_layer):
                    break
                i = i + 1

        loss = torch.zeros(1, requires_grad=True)

        for pair in content_feature:
            loss = loss + self.content_criterion(pair[0], pair[1]) * self.content_w

        for pair in style_feature:
            loss = loss +  self.content_criterion(pair[0], pair[1])\
                    * self.style_w / (4 * (pair[2]**2) * (pair[3]**2))
                    
        return loss

class ImageStyleTransfer():
    def __init__(self, vgg = models.vgg19(pretrained=True), content_weight = 1,
                 style_weight = 1000, content_features_layer = dft_content_features_layer,
                 style_features_layer = dft_style_features_layer, step_size=100):
        
        self.model = model(vgg, content_weight, style_weight, content_features_layer,
                           style_features_layer)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model.to(device)
        self.device = device
        self.step_size = step_size
        
    def merge(self, content, style):
        x = torch.randn_like(content, requires_grad = True)
        x.to(self.device)
        content.to(self.device)
        style.to(self.device)
        optimizer = optim.LBFGS([{'params':x}])
        
        print("Start merging...")
        for i in range(self.step_size):
            def closure():
                optimizer.zero_grad()
                loss = self.model(x, content, style)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            if i % 10 == 0:
                loss = self.model(x, content, style)
                print("After %d iterations, loss = %f"%(i, loss))
        
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--content', required=True)
    parser.add_argument('-s','--style', required=True)

    # get arguments
    io_args = parser.parse_args()
    content_pic_name = io_args.content
    style_pic_name = io_args.style
    
    # read pictures
    content_mat = cv.imread(content_pic_name).astype(np.float32)
    style_mat = cv.imread(style_pic_name).astype(np.float32)

    # TODO make sure content and style are of same sizes
    height = min(content_mat.shape[1], style_mat.shape[1])
    width = min(content_mat.shape[2], style_mat.shape[2])
    content_mat = CenterCrop(content_mat, height, width)
    style_mat = CenterCrop(style_mat, height, width)

    # transform picture matrices for vgg to hangle
    content_mat = content_mat / 255
    style_mat = style_mat / 255    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    content_mat = transform(content_mat)
    style_mat = transform(style_mat)
    content_mat = content_mat.unsqueeze(0)
    style_mat = style_mat.unsqueeze(0)
    content_mat.requires_grad = False
    style_mat.requires_grad = False
    
    # merge content and style
    merger = ImageStyleTransfer(style_weight=10000, step_size = 20)
    x = merger.merge(content_mat, style_mat)
    
    # save and show x
    x = x.squeeze()
    detransform = transforms.Compose([transforms.Normalize([-0.485 / 0.229, -0.456/0.224,-0.406/0.225],
                                                           [1./0.229,1./0.224,1./0.225])])
    pickle.dump(x, open("x.dat","wb"))
    x = detransform(x.detach())
    xnp = x.numpy()
    # during training x might be out of valid range, so make it more robust
    xnp[xnp<0] = 0
    xnp[xnp>1] = 1
    #print(xnp)
    plt.imshow(np.transpose(xnp,(1,2,0)))
    plt.savefig(content_pic_name + style_pic_name + '.jpg')
    
    
