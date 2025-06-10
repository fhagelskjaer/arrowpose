#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

class WeightedLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.alpha = torch.tensor([0.3893, 1.7058, 0.9049]).cuda()

    def forward(self, inputs, targets):
        targets = targets.contiguous().view(-1)

        loss = F.cross_entropy(inputs, targets, reduction='none')
        
        targets = targets.type(torch.long)
        targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        # print( at.size(), loss.size(), inputs.size() )
        F_loss = at*loss
        return F_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        # logpt = F.log_softmax(input)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class StandardLoss(nn.Module):
    def __init__(self, size_average=True):
        super(StandardLoss, self).__init__()
        self.size_average = size_average
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, input, target):
        target = target.contiguous().view(-1)
        loss = F.cross_entropy(input, target, reduction='none')
        return loss.mean()

class ClassicLoss(nn.Module):
    def __init__(self, gammas=[0.33,0.33,0.33], size_average=True):
        super(ClassicLoss, self).__init__()
        self.size_average = size_average
        self.mae_loss = nn.L1Loss(reduction='none')
        self.cls_weight = torch.tensor(gammas).cuda()
        self.dist_weight = torch.tensor([0,1]).cuda()


    def forward(self, seg_pred, seg, cen_pred, cen, top_pred, top, vw):
        seg = seg.contiguous().view(-1)
        cw = self.cls_weight.gather(0,seg)
        loss = F.cross_entropy(seg_pred, seg, reduction='none') * cw * vw
        focal_loss = loss.mean()
        
        ow = seg>0
        ow = ow.type(torch.float32)
        ow = ow.view(-1)

        # loss1 = self.mae_loss(cen_pred, cen).mean(dim=1)*ow*cw
        per_instance_dist_loss = cen - cen_pred # Abs,
        # print( per_instance_dist_loss.size() )
        per_instance_dist_loss = per_instance_dist_loss.abs().mean(dim=-1)
        per_instance_dist_loss = per_instance_dist_loss * ow
        per_instance_dist_loss = per_instance_dist_loss * cw * vw
        loss1 = per_instance_dist_loss.mean()

        # loss2 = self.mae_loss(top_pred, top).mean(dim=1)*ow*cw
        per_instance_top_loss = top - top_pred # abs,
        per_instance_top_loss = per_instance_top_loss.abs().mean(dim=-1)
        per_instance_top_loss = per_instance_top_loss * ow
        per_instance_top_loss = per_instance_top_loss * cw * vw
        loss2 = per_instance_top_loss.mean()

        # print(focal_loss, loss1.mean(), loss2.mean())
        loss = focal_loss*0.33 + loss1.mean()*0.33 + loss2.mean()*0.33
        return loss




class FocalEtHuberLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalEtHuberLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.mae_loss = nn.L1Loss(reduction='none')
        self.obj_weight = torch.tensor([0, 1]).cuda()


    def forward(self, input, target, cen_pred, cen, top_pred, top):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        # logpt = F.log_softmax(input)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        focal_loss = loss.mean()
        
        ow = target>0
        ow = ow.type(torch.float32)
        ow = ow.view(-1)

        l1_loss1 = self.mae_loss(cen_pred, cen).sum(dim=1)*ow
        l1_loss2 = self.mae_loss(top_pred, top).sum(dim=1)*ow

        loss = focal_loss + l1_loss1.mean() + l1_loss2.mean()
        return loss


        # if self.size_average: return loss.mean()
        # else: return loss.sum()

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        # self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.alpha = torch.tensor([alpha, 1-alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.contiguous().view(-1)
        # print( inputs.size() )
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # print( BCE_loss.size() )
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        # targets_loss = targets > 0
        # targets_loss.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        # print( at.size(), pt.size() )
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        # print( F_loss.mean().size() )
        return F_loss.mean()

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
