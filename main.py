#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday April 20 2024
@author: dinesh
This gitHub repo contains the implementation of the RiT layer and an example of it can be embedded with
the ResNet18 model.

See the following papers for implementation details:
    
Kumar, D., & Sharma, D. (2023). Feature map augmentation to improve scale invariance in convolutional 
neural networks. Journal of Artificial Intelligence and Soft Computing Research, 13(1), 51-74.    

Kumar, D., Sharma, D., & Goecke, R. (2020, February). Feature map augmentation to improve rotation 
invariance in convolutional neural networks. In International Conference on Advanced Concepts for 
Intelligent Vision Systems (pp. 348-359). Cham: Springer International Publishing.
"""

import torch
import time
import torch.nn as nn
import torch.optim as optim

#imports from custom module
from Model_RiT import Model_RESNET18_RiT, BasicBlock
from Dataset import LoadImageHoof_Torch

seed = 10
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_default_dtype(torch.float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#%%
def train(net, criterion, optimizer, trainloader, train_cache):  
    start = time.time()
    
    (EPOCHS,momnt,lr,w_decay, loss_fn, learning_fn) =train_cache
    
    net.train()
    for epoch in range(0, EPOCHS): #of iterations
        epochstart = time.time()

        running_loss = 0.0
        tot_acc = 0.0
        
        for i, data in enumerate(trainloader, 0):
  
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)   #automatically does the forward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
                       
            _, trpred = torch.max(outputs.data, 1)
            acc = (trpred == labels).sum().item()
            acc = acc/labels.size(0)
            tot_acc += acc

            print('\r', 'epoch: %d | batch: %5d | loss: %.3f | acc: %.3f | time: %.4f' % 
                      (epoch + 1, i + 1, running_loss / (i+1), tot_acc/(i+1), time.time()-epochstart), end='\r', flush=True)
                      

        epoch_loss = running_loss/(len(trainloader.dataset)/trainloader.batch_size)
        epoch_acc = tot_acc/(len(trainloader.dataset)/trainloader.batch_size)
                 
        print ('\n')
        tot_time = time.time() - start
             
    tot_time = time.time() - start
    print ('Training completed in (seconds) %.3f ' %(tot_time))
    
    return {'loss': epoch_loss,\
            'acc': epoch_acc}

#%%
def test(net, testloader):  
    correct = 0
    total = 0
    firstpass = True
    with torch.no_grad():
        i = 0
        for data in testloader:
            #images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            #concat predictions
            if firstpass:
                y_preds = predicted
                firstpass = False
            else:
                y_preds = torch.cat((y_preds, predicted), 0)
            i = i+1
            print('\r', 'processing image: %d ' % 
                      (i + 1), end='\r', flush=True)


    test_acc = (correct / total) 
    if device.type == 'cuda':              
        _preds = y_preds.cpu().tolist()
    else:
        _preds = y_preds.tolist()
    
    return {'acc': test_acc,\
            'labels': testloader.dataset.targets,\
            'predictions': _preds} 
            
#%%
if __name__== '__main__':

    #get the dataset      
    trainset, trainloader, testset, testloader = LoadImageHoof_Torch()
    
    #setup basic training parameters
    EPOCHS = 1
    lr = 0.0001
    momnt = 0.9
    w_decay = 0.0001
    
    
    loss_fn = 'CrossEntropy' #try different loss functions
    learning_fn = 'SGD' #try different learning functions eg. 'ADAM'
    train_cache = (EPOCHS,momnt,lr,w_decay, loss_fn, learning_fn)

    #create the ResNet18 model with embedded RiT layer.           
    net = Model_RESNET18_RiT('resnet18', BasicBlock, [2, 2, 2, 2], num_classes=10, 
                                      pretrained=True, progress=True, freezeFeatureWeights=False, 
                                      freezeClassifierWights=False)    
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momnt, weight_decay=w_decay)
       
    #train
    tr_hist = train(net, criterion, optimizer, trainloader, train_cache)
    tr_loss = tr_hist['loss']
    tr_acc = tr_hist['acc']
    print ('Final train loss: %.3f | acc: %.3f' % (tr_loss,tr_acc))
        
    #test accuracy
    net.eval()   
    test_hist = test(net, testloader)
    print('Accuracy of the network on the test images: %.3f %%' % (test_hist['acc']*100))

    
