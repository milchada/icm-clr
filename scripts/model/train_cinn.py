import torch
import torch.nn
import torch.optim
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm

from . import combined_model
from . import losses
from scripts.model.experiment_tracking import NeptuneExperimentTracking, VoidExperimentTracking
from scripts.model.optimizer import Optimizer
from scripts.model.training import Trainer, run_training
from scripts.data import data
from scripts.data.augmentations import SimCLRAugmentation

import config as c

import os

#Load Parameters
import yaml
params = yaml.safe_load(open('params.yaml'))

neptune_params = params['neptune']
train_default_params = params['train_cinn']

def loss_2_host(x):
    return x.cpu().detach().numpy()

def loss_dict_2_host(d):
    return dict(map(lambda x: (x[0], loss_2_host(x[1])), d.items()))

def train_cinn(params={},
               model_index=0,
               save_model=True,
               save_path=None,
               use_checkpoint=False,
               optuna_trial=None,
               experiment_tracking=True):
    """
    Function to set up and train the cINN

    :param params: dictionary with training and model parameter; fields which are not defined are taken from params.yaml
    :param model_index: define the index of the model for ensemble averaging
    :param save_model: if true metrics and model are saved; otherwise only the validation losses are returned (for parameter optimization)
    :param experiment_tracking: switch for experiment tracking by neptune.ai (has to be set up in params.yaml)
    
    """ 
    

    #Use default parameters if not set
    params = dict(train_default_params, **params)
    
    #Init experiment tracking
    if experiment_tracking:
        experiment_tracker = NeptuneExperimentTracking(params, tags=['cinn'])
    else: 
        experiment_tracker = VoidExperimentTracking()
            
    #Load and prepare Model
    model = combined_model.CombinedModel(params)
    
    if params["FIX_RESNET_PARAMS"]:
        model.fix_resnet_weights()
    
    if params["USE_PRETRAINED_RESNET"]:
        model.load_pretrained_resnet(c.resnet_path)
        
    model.to(c.device)
    
    #Init the optimizer
    optimizer = Optimizer(model, params)
    
    #Set the save path of the model
    if save_model:
        if not isinstance(save_path, str):
            save_path = c.cinn_path(model_index)
    else:
        save_path = None
    
    #Init the trainer
    trainer = Trainer(model,
                      optimizer,
                      experiment_tracker,
                      params["PATIENCE"],
                      params["NUM_EPOCHS"],
                      save_path,
                      max_num_batches = params["MAX_NUM_BATCHES_PER_EPOCH"],
                      max_runtime_seconds = params["MAX_RUNTIME_SECONDS"],
                      use_checkpoint=use_checkpoint,
                      optuna_trial=optuna_trial)
    
    #Prepare the data
    augmentation = SimCLRAugmentation(params["AUGMENTATION_PARAMS"])
    training_data = data.get_train_loader(batch_size=params["BATCH_SIZE"], labels=True, augmentation=augmentation, n_views=1, shuffle=True, drop_last=True) 
    validation_data = data.get_val_loader(batch_size=params["BATCH_SIZE"], labels=True, augmentation=None, n_views=1, shuffle=True, drop_last=True)
    
    #Training/Evaluation Functions
    #-----------------------------------------------------------------------------
    
    def training_lossfunction(model, batch):

        print(batch)
        
        image, label = batch[0]
        image = torch.cat(image, dim=0)
        image = image.to(c.device)
        label = label.to(c.device)

        #Augmentation
        label += params["NOISE"] * torch.randn_like(label)
                    
        #Forward Pass
        z, log_j = model(label, image)  

        #Init loss list
        loss = []
        loss_dict={}
        
        #Negative log likelihood
        if losses.lambd_max_likelihood != 0:
            nll = losses.loss_max_likelihood(z, log_j)
            loss_dict = {**loss_dict, 'Train_Loss_NLL': nll}
            loss.append(nll * losses.lambd_max_likelihood)

        #Forward MMD
        #Ensure that z is multivariate Gaussian
        if losses.lambd_mmd_forw != 0:
            fmmd = losses.loss_forward_mmd(z)
            loss_dict = {**loss_dict, 'Train_Loss_FMMD': fmmd}
            loss.append(fmmd * losses.lambd_mmd_forw)

        #Backward MMD
        #Ensure that x prior distribution is reproduced
        if losses.lambd_mmd_back != 0:
            x_samples = model.reverse_sample(torch.randn_like(z), image)
            bmmd = losses.loss_backward_mmd(label, x_samples)
            loss_dict = {**loss_dict, 'Train_Loss_BMMD': bmmd}
            loss.append(bmmd * losses.lambd_mmd_back)

        #MSE l -> x
        #Just as for the MLP model
        if losses.lambd_mse != 0:
            x_zero = model.reverse_sample(torch.zeros_like(z), image)
            mse = losses.loss_mse(label, x_zero)
            loss_dict = {**loss_dict, 'Train_Loss_MSE': mse}
            loss.append(mse * losses.lambd_mse)

        #MAE l -> x
        #Just as for the MLP model
        if losses.lambd_mae != 0:
            x_zero = model.reverse_sample(torch.zeros_like(z), image)
            mae = losses.loss_mae(label, x_zero)
            loss_dict = {**loss_dict, 'Train_Loss_MAE': mae}
            loss.append(mae * losses.lambd_mae)

        #Use all losses (!=0) to perform the error propagation
        loss = sum(loss)
        loss_dict = {**loss_dict, 'Train_Loss': loss}

        loss_dict = loss_dict_2_host(loss_dict)
        
        return loss, loss_dict

    
    def validation_lossfunction(model, batch):
    
        image, label = batch[0]
        image = torch.cat(image, dim=0)
        image = image.to(c.device)
        label = label.to(c.device)
        
        #Forward Pass
        z, log_j = model(label, image) 
        
        #Init loss list
        loss = []
        loss_dict={}
        
        #Negative log likelihood
        if losses.lambd_max_likelihood != 0:
            nll = losses.loss_max_likelihood(z, log_j)
            loss_dict = {**loss_dict, 'Val_Loss_NLL': nll}
            loss.append(nll * losses.lambd_max_likelihood)

        #Forward MMD
        #Ensure that z is multivariate Gaussian
        if losses.lambd_mmd_forw != 0:
            fmmd = losses.loss_forward_mmd(z)
            loss_dict = {**loss_dict, 'Val_Loss_FMMD': fmmd}
            loss.append(fmmd * losses.lambd_mmd_forw)

        #Backward MMD
        #Ensure that x prior distribution is reproduced
        if losses.lambd_mmd_back != 0:
            x_samples = model.reverse_sample(torch.randn_like(z), image)
            bmmd = losses.loss_backward_mmd(label, x_samples)
            loss_dict = {**loss_dict, 'Val_Loss_BMMD': bmmd}
            loss.append(bmmd * losses.lambd_mmd_back)

        #MSE l -> x
        #Just as for the MLP model
        if losses.lambd_mse != 0:
            x_zero = model.reverse_sample(torch.zeros_like(z), image)
            mse = losses.loss_mse(label, x_zero)
            loss_dict = {**loss_dict, 'Val_Loss_MSE': mse}
            loss.append(mse * losses.lambd_mse)

        #MAE l -> x
        #Just as for the MLP model
        if losses.lambd_mae != 0:
            x_zero = model.reverse_sample(torch.zeros_like(z), image)
            mae = losses.loss_mae(label, x_zero)
            loss_dict = {**loss_dict, 'Val_Loss_MAE': mae}
            loss.append(mae * losses.lambd_mae)

        #Also save statistics about the z
        loss_dict = {**loss_dict, 'Val_z_Mean': torch.max(torch.abs(torch.mean(z, 0)))}
        loss_dict = {**loss_dict, 'Val_z_Std': torch.max(torch.std(z, 0))}
        
        #Use all losses (!=0) to perform the error propagation
        loss = sum(loss)
        loss_dict = {**loss_dict, 'Val_Loss': loss}
        
        loss_dict = loss_dict_2_host(loss_dict)
        loss = loss_2_host(loss)

        return loss, loss_dict
                    
        #Check if RESNET params should be unfixed
        #    	if params["FIX_RESNET_PARAMS"] and optimizer.lr < params["RESNET_LR_THRESHOLD"]:
        #        params["FIX_RESNET_PARAMS"] = False
        #        cinn.unfix_resnet_weights()
        #        optimizer.update(keep_lr=True)
        #        print("Unfix RESNET")

    #Perform training
    return run_training(trainer, training_lossfunction, [training_data], validation_lossfunction, [validation_data])


if __name__ == "__main__":
    
    NUM_MODELS = train_default_params["NUM_MODELS"]
    
    for i in range(NUM_MODELS):
        train_cinn(model_index=i)
        