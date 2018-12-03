# -*- coding: utf-8 -*-
"""
Author: Zongyue Qin
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import os
import time
import copy
from matplotlib.image import imread
import argparse
import cv2 as cv
import pickle

class StyleTransfer:
    
    def __init__(self, step_size = 0.001):
       self.vgg =  models.vgg11(pretrained=True)
       for param in self.vgg.parameters():
           param.requires_grad = False
 
       self.vgg.classifier = nn.Sequential(nn.ReLU(True)) # The classifier will give output exact same as input
       self.step_size = step_size
    
    # merge content and style
    def merge(self, content, style):
       
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.vgg.to(device)
        
        # generate x randomly
        x = torch.randn_like(content).unsqueeze(0)
        x = x * 0.001
        x.requires_grad_(True)
        x.to(device)
        
        # extract features from content
        content_output =  self.vgg(content.unsqueeze(0).to(device))
        content_output.detach()

        criterion = nn.MSELoss()
#        optimizer = optim.SGD([{'params':x}], lr=self.step_size, momentum=0.9)   
        optimizer = optim.LBFGS([{'params':x}])

        
        
        # training x        
        for i in range(1000):
            def closure():
                optimizer.zero_grad()
                output = self.vgg(x)
                loss_content = criterion(output, content_output)
                loss_content.backward()
                return loss_content
            
            optimizer.step(closure)
            #self.vgg16.zero_grad()
#            optimizer.zero_grad()
#            output = self.vgg16(x)
#            loss_content = criterion(output, content_output)
        
#            loss_content.backward()
            #x.data = x.data - self.step_size * x.grad
#            optimizer.step()
            if i % 100 == 0:
                output = self.vgg(x)
                loss_content = criterion(output, content_output)
                print("%d iterations, loss = %f"%(i, loss_content))
            
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
    
    # transform picture matrices for vgg to hangle
    content_mat = content_mat / 255
    style_mat = style_mat / 255    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    content_mat = transform(content_mat)
    style_mat = transform(style_mat)
    
    # merge content and style
    transfer = StyleTransfer(step_size = 1)
    x = transfer.merge(content_mat, style_mat)
    
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
    print(xnp)
    plt.imshow(np.transpose(xnp,(1,2,0)))
    plt.savefig('generate.jpg')
    
    