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
mmd_kernels = losses_params['mmd_kernels']
mmd_kernel_type = losses_params['mmd_kernel_type']

adaption_type = losses_params['adaption_type']

lambd_simclr_train = losses_params['lambd_simclr_train']
lambd_simclr_domain = losses_params['lambd_simclr_domain']
lambd_simclr_adaption = losses_params['lambd_simclr_adaption']

lambd_max_likelihood = losses_params['lambd_max_likelihood']
lambd_mmd_forw = losses_params['lambd_mmd_forw']
lambd_mmd_back = losses_params['lambd_mmd_back']
lambd_mse = losses_params['lambd_mse']
lambd_mae = losses_params['lambd_mae']

nce_temperature = losses_params['nce_temperature']

def MMD_kernel(x, exponents):
    if mmd_kernel_type == 'inverse_multiquadratic':
        C = exponents[0]
        a = exponents[1]
        return C**a * ((C + x) / a)**-a
    elif mmd_kernel_type == 'multiscale':
        a = exponents
        return a**2 * (a**2 + x)**-1
    elif mmd_kernel_type == 'gaussian_rbf':
        a = exponents
        return torch.exp(-(x/a)**2)
    else:
        raise ValueError("MMD Kernel Unknown!")

def l2_dist_matrix(x, y):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    return dxx, dyy, dxy          
        
def MMD_matrix(x, y, exponents):
    dxx, dyy, dxy = l2_dist_matrix(x, y)

    XX, YY, XY = (torch.zeros(dxx.shape).cuda(),
                  torch.zeros(dyy.shape).cuda(),
                  torch.zeros(dxy.shape).cuda())

    for e in exponents:
        XX += MMD_kernel(dxx, e)
        YY += MMD_kernel(dyy, e)
        XY += MMD_kernel(dxy, e)

    return XX + YY - 2.*XY

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
    loss = torch.nn.CrossEntropyLoss().to(c.device)(logits, labels)
    return loss, logits, labels 

def forward_mmd(y0, y1):
    return MMD_matrix(y0, y1, mmd_forw_kernels)

def backward_mmd(x0, x1):
    return MMD_matrix(x0, x1, mmd_back_kernels)

def mmd(x0, x1):
    return MMD_matrix(x0, x1, mmd_kernels)

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

def loss_mmd(x, y):
    return torch.mean(mmd(x, y))

def loss_linear_mmd(x, y):
    dxx = torch.cdist(x, x, p=1)
    dyy = torch.cdist(y, y, p=1)
    dxy = torch.cdist(x, y, p=1)
    
    XX, YY, XY = (torch.zeros(dxx.shape).cuda(),
                  torch.zeros(dyy.shape).cuda(),
                  torch.zeros(dxy.shape).cuda())

    for e in mmd_kernels:
        XX += MMD_kernel(dxx, e)
        YY += MMD_kernel(dyy, e)
        XY += MMD_kernel(dxy, e)
    
    return torch.mean(XX + YY - 2.*XY)

def loss_kld(x, y):
    kl_loss = torch.nn.KLDivLoss(reduction="sum", log_target=True)
    log_x = F.log_softmax(x)
    log_y = F.log_softmax(y)
    return kl_loss(log_x, log_y)
    
def loss_huber(loss, delta=1):
    if torch.abs(loss) < delta:
        return 0.5*loss**2
    else:
        return delta * (torch.abs(loss) - 0.5*delta)
    
def loss_adaption(x, y):
    if adaption_type == 'mmd':
        return loss_mmd(x, y)
    elif adaption_type == 'linear_mmd':
        return loss_linear_mmd(x, y)
    elif adaption_type == 'kld':
        return loss_kld(x, y)
    elif adaption_type == 'huber_mmd':
        loss = loss_mmd(x, y)
        return loss_huber(loss)
    elif adaption_type == 'huber_linear_mmd':
        loss = loss_linear_mmd(x, y)
        return loss_huber(loss)
    elif adaption_type == 'huber_kld':
        loss = loss_kld(x, y)
        return loss_huber(loss)
    else:
        raise ValueError("Adaption Loss Unknown!") 
        
    
        
    
