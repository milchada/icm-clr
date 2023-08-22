import optuna
import torch
from torch.cuda.amp import autocast
import numpy as np
import config as c
import os
from tqdm import tqdm
from time import time
from scripts.util.logging import logger
from scripts.util.make_dir import make_dir

class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 experiment_tracker,
                 patience,
                 num_epochs,
                 save_path,
                 max_num_batches=None,
                 max_runtime_seconds=None,
                 use_checkpoint=False,
                 optuna_trial=None):
        
        self._epoch = 0
        self._val_loss_memory = []
        
        self._patience = patience
        self._num_epochs = num_epochs
        self._save_path = save_path
        self._max_num_batches = max_num_batches
        self._max_runtime_seconds = max_runtime_seconds
        self._time_start = time()
        self._optuna_trial = optuna_trial

        self.model = model
        self.optimizer = optimizer
        self.experiment_tracker = experiment_tracker
        
        if use_checkpoint:
            self.load()
    
    @property
    def epoch(self):
        return self._epoch 

    @property
    def runtime_seconds(self):
        return (time() - self._time_start)

    def optuna_report(self, intermediate_value):
        if self._optuna_trial is not None:
            #To keep the pruning independent from the batch size we report as #step# the runtime in bins of 20min
            self._optuna_trial.report(intermediate_value, int(self.runtime_seconds/1200))

    def optuna_should_prune(self):
        if self._optuna_trial is not None:
            if self._optuna_trial.should_prune():
              raise optuna.TrialPruned()

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
        
        assert isinstance(datasets, list)
        
        self.model.train()        
        
        batch_num = 0
        
        for batch in zip(*datasets):
            
            #Run only a max number of batches per epoch
            if batch_num < self._max_num_batches:
                batch_num += 1
            else:
                break
            
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
        
        assert isinstance(datasets, list)
        
        self.model.eval()
        
        loss_list = []
        lossdict_list = []
        
        batch_num = 0
        
        for batch in zip(*datasets):
            
            #Run only a max number of batches per epoch
            if batch_num < self._max_num_batches*0.125 or batch_num < 10:
                batch_num += 1
            else:
                break
            
            with autocast(enabled=True), torch.cuda.device(c.device), torch.no_grad():
                loss, lossdict = lossfunction(self.model, batch)
                loss_list.append(loss)
                lossdict_list.append(lossdict)
            
        loss_mean = np.mean(loss_list)
        lossdict_mean = self._dict_mean(lossdict_list)
         
        self.experiment_tracker.log_metric(lossdict_mean) 
        self._val_loss_memory.append(loss_mean)
        self.optimizer.lr_step(loss_mean)
        
        self.optuna_report(loss_mean)

        return loss_mean, lossdict_mean
    
    def early_stopping_criterion(self):
        if len(self._val_loss_memory) > 0:
            epoch_min_val =  np.argmin(self._val_loss_memory)
            return (self.epoch + 1) > (epoch_min_val + self._patience)
        else:
            return False
            
    def max_epochs_criterion(self):
        return self.epoch >= (self._num_epochs - 1)
    
    def max_time_criterion(self):
        if self._max_runtime_seconds is not None:
            return self.runtime_seconds >= self._max_runtime_seconds
        else:
            return False

    def stopping_criterion(self):
        '''Test stopping criteria'''
        return self.early_stopping_criterion() or self.max_epochs_criterion() or self.max_time_criterion()

    def save(self):
        '''Save model'''
        if self._save_path is not None:
            make_dir(self._save_path)
            torch.save(self.model.state_dict(), self._save_path)
            logger.info('Model saved.')
            
    def load(self):
        '''Load model'''
        if self._save_path is not None and os.path.isfile(self._save_path):
            checkpoint = torch.load(self._save_path, map_location=torch.device(c.device))
            self.model.load_state_dict(checkpoint)
            logger.info('Existing Model loaded.')
        
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

        #Prune the optuna run if it is marked
        self.optuna_should_prune()
        
        #Next epoch
        self.step()
        return self.epoch


def run_training(trainer, training_lossfunction, training_data, validation_lossfunction, validation_data):
    '''Perform the training and return the final validation loss'''
    
    assert isinstance(trainer, Trainer)

    logger.info('Start training...')
    for epoch_counter in tqdm(trainer):
        
        trainer.perform_training_step(training_lossfunction, training_data)
        val_loss, _ = trainer.perform_validation_step(validation_lossfunction, validation_data)  
            
    return val_loss
