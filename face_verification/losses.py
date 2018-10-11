from __future__ import division, print_function
import numpy as np
import torch
import cv2
from torch import nn


def loss(pred, gt):
    # return nn.BCELoss().cuda()(pred, gt)
    return nn.BCELoss().cuda()(pred, gt)


def accuracy(pred, gt):
    # return torch.mean((torch.round(pred[gt != -1]) == gt[gt != -1]).float())
    return torch.mean((torch.round(pred) == gt).float())
