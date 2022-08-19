import torch
from torch.cuda.amp import autocast
import numpy as np
import config as c
import os

class Trainer(object):
    def __init__(self, model, optimizer, experiment_tracker, patience, num_epochs, save_path):
        
        self._epoch = 0
        self._val_loss_memory = []
        
        self._patience = patience
        self._num_epochs = num_epochs
        self._save_path = save_path
        
        self.model = model
        self.optimizer = optimizer
        self.experiment_tracker = experiment_tracker
    
    @property
    def epoch(self):
        return self._epoch

    def step(self):
        self._epoch += 1
        
    def __len__(self):
        return self._num_epochs
    
    def perform_training_step(self, lossfunction, datasets):
        """
        Perform a training step
        
        Args:
            lossfunction (callable): Lossfunction which the signature 
                                     (pytorchmodel, batches of images) -> training loss as torchtensor in device memory
                                     (i.e. with trainable gradient) and a lossdict with losses as floats in host memory
            datasets (list):         List of datasets to use for training
        """
        
        self.model.train()        
        
        for batch in zip(*datasets):
            
            with autocast(enabled=True), torch.cuda.device(c.device):
                loss, lossdict = lossfunction(self.model, batch)
            
            self.experiment_tracker.log_metric(lossdict)
            self.experiment_tracker.log_metric({'learning_rate': self.optimizer.lr})
            
            #Clip Gradient to prevent too large gradients 
            torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters, 1.)
            
            #Perform a training step
            self.optimizer.backward(loss)
            
        return loss, lossdict
    
    def _dict_mean(self, x):
        '''Calculate the mean of values across a list of dicts'''
        
        l = []
        
        for lossdict in x:
            l.append(np.array(list(lossdict.values())))
        
        mean = np.mean(l, axis=0)
        
        return dict(zip(list(x[0].keys()), mean))
        
    
    def perform_validation_step(self, lossfunction, datasets):
        """
        Perform a validation step
        
        Args:
            lossfunction (callable): Lossfunction which the signature 
                                     (pytorchmodel, batches of images) -> validation loss as float in host memory
                                     and a lossdict with losses as floats in host memory
            datasets (list): List of datasets to use for validation
        """
        self.model.eval()
        
        loss_list = []
        lossdict_list = []
        
        for batch in zip(*datasets):
            
            with autocast(enabled=True), torch.cuda.device(c.device), torch.no_grad():
                loss, lossdict = lossfunction(self.model, batch)
                loss_list.append(loss)
                lossdict_list.append(lossdict)
            
        loss_mean = np.mean(loss_list)
        lossdict_mean = self._dict_mean(lossdict_list)
         
        self.experiment_tracker.log_metric(lossdict_mean) 
        self._val_loss_memory.append(loss)
        self.optimizer.lr_step(loss)
        
        return loss_mean, lossdict_mean
    
    def early_stopping_criterion(self):
        if len(self._val_loss_memory) > 0:
            epoch_min_val =  np.argmin(self._val_loss_memory)
            return (self.epoch + 1) > (epoch_min_val + self._patience)
        else:
            return False
            
    def max_epochs_criterion(self):
        return self.epoch >= (self._num_epochs - 1)
    
    def stopping_criterion(self):
        '''Test stopping criteria'''
        return self.early_stopping_criterion() or self.max_epochs_criterion()
    
    def save(self):
        '''Save model'''
        if not os.path.exists(c.model_path):
            os.makedirs(c.model_path)

        if self._save_path is not None:
            torch.save(self.model.state_dict(), self._save_path)
        
    def __iter__(self):
        self._epoch = 0
        return self
        
    def __next__(self):
        '''When used as iterator object: save and check stop criterions each iteration'''
        
        #Save model
        self.save()
        
        #Check if stopping criterions are reached
        if self.stopping_criterion():
            raise StopIteration
        
        #Next epoch
        self.step()
        return self.epoch