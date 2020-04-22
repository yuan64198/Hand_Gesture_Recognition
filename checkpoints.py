# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:41:21 2020

@author: Chih-Yuan Huang
"""

import os
import torch

def save_checkpoint( state, every):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    if state["epoch"] % every == 0:
        torch.save(state, "checkpoints/checkpoint_{}.pth".format(str(state["epoch"]).zfill(3)))

def load_checkpoint(path, model, optimizer):
    start_epoch = 0
    if path:
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                path, checkpoint['epoch']))
    return start_epoch