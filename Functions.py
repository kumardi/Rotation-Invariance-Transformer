#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday April 20 2024
Here we implement the RiT functions for ResNet18 model
@author: dinesh
"""

import torch
from torch.autograd import Function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
class Function_RiT(Function):    
  
    @staticmethod
    def forward(ctx, input):      
        ctx.save_for_backward(input)
        
        output = input.clone()
        
        x90 = output.transpose(2, 3)
        x270 = output.transpose(2, 3).flip(3)

        output = torch.cat((output, x90), 1)          
        output = torch.cat((output, x270), 1)

        return (output)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None # set output to None
        
        a, = ctx.saved_tensors
        
        grad_input = grad_output.clone()   
               
        adv=0
        outputA = grad_input[:, adv:adv+a.shape[1],:,:]
        return outputA

#%%%%        
class Function_Flatten(Function):
    @staticmethod
    def forward(ctx, x, y):
        outputX = x.clone()
        outputY = y.clone()
        
        outputX = outputX.view(outputX.shape[0], -1) 
        outputY = outputY.view(outputY.shape[0], -1) 
        
        ctx.save_for_backward(x,y) # save input for backward pass
        
        return torch.cat((outputX, outputY), 1)          
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None # set output to None
        x, y = ctx.saved_tensors # restore input from context       
        grad_input = grad_output.clone()
        
        dimx = x.shape[1] * x.shape[2] * x.shape[3]
        outputX = grad_input[:, 0:dimx]
        outputY = grad_input[:, dimx:grad_input.shape[1]]
        
        outputX = outputX.view(x.shape)
        outputY = outputY.view(y.shape)

        return outputX, outputY
