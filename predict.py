# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:29:41 2020

@author: Chih-Yuan Huang
"""
import torch
import numpy as np
from model import Gest_CNN
from checkpoints import load_checkpoint

MAX_LEN = 51

def predict( arr, checkpoint_path):
    padding = np.zeros((3, MAX_LEN))
    padding[:, :len(arr[0])] = arr
    arr = padding
    model = Gest_CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    _ = load_checkpoint(checkpoint_path, model, optimizer)
    tensor = torch.from_numpy(np.asarray([padding])).float()
    output = model(tensor)
    pred = output.argmax(dim=1, keepdim=True)
    return pred.data.numpy()[0][0]+1
    