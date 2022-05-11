# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:05:28 2021

Loss functions for model training and evaluation

@author: lukas
"""

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import config as c

params = yaml.safe_load(open('params.yaml'))
losses_params = params['losses']
mmd_forw_kernels = losses_params['mmd_forw_kernels']
mmd_back_kernels = losses_params['mmd_back_kernels']

lambd_max_likelihood = losses_params['lambd_max_likelihood']
lambd_mmd_forw = losses_params['lambd_mmd_forw']
lambd_mmd_back = losses_params['lambd_mmd_back']
lambd_mse = losses_params['lambd_mse']
lambd_mae = losses_params['lambd_mae']

nce_temperature = losses_params['nce_temperature']

def MMD_matrix_multiscale(x, y, widths_exponents):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda())

    for C,a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return XX + YY - 2.*XY

def l2_dist_matrix(x, y):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    return torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

def info_nce_loss(features, n_views, batch_size):

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(c.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(c.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(c.device)

    logits = logits / nce_temperature
    return logits, labels

def loss_simclr(x, n_views, batch_size):
    logits, labels = info_nce_loss(x, n_views, batch_size)
    return torch.nn.CrossEntropyLoss().to(c.device)(logits, labels), logits, labels 

def forward_mmd(y0, y1):
    return MMD_matrix_multiscale(y0, y1, mmd_forw_kernels)

def backward_mmd(x0, x1):
    return MMD_matrix_multiscale(x0, x1, mmd_back_kernels)

def loss_mse(x0, x1):
    return torch.mean((x0 - x1)**2)

def loss_mae(x0, x1):
    return torch.mean(torch.abs(x0 - x1))
    
def loss_max_likelihood(z, log_j):
    return torch.mean(0.5 * torch.sum(z**2, 1) - log_j)

def loss_forward_mmd(z):
    return torch.mean(forward_mmd(z, torch.randn_like(z)))

def loss_backward_mmd(x, x_samples):
    return torch.mean(backward_mmd(x, x_samples))
