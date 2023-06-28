'''

Script to optimize the model using optuna

'''

import optuna
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileSymlinkLock

import os
import torch
import gc


from scripts.model.train_clr import train_clr
from scripts.util.make_dir import make_dir
from scripts.postprocessing.write_representation import write_representation
import config as c

import yaml

#Load the default params from the parameter file
opt_params = yaml.safe_load(open('params.yaml'))['params_opt']

class ParameterOptimization:
    
    def __init__(self):
        pass
    
    def _params(self, trail):
        pass
    
    def _objective(self, trail):
        pass
        
    def _create_study(self):
        optuna_storage_lock = JournalFileSymlinkLock(c.optuna_storage)
        optuna_storage = JournalStorage(JournalFileStorage(c.optuna_storage, lock_obj=optuna_storage_lock))
        self.study = optuna.create_study(directions=self.direction,
                                         load_if_exists=True,
                                         storage=optuna_storage,
                                         study_name=opt_params['STUDY_NAME'])
        
    def run(self):
        self._create_study()
        self.study.optimize(self._objective, gc_after_trial=True, n_trials=opt_params['NUM_TRIALS'], catch=(torch.cuda.OutOfMemoryError))
   

class ParameterOptimizationCLR(ParameterOptimization):
    
    def __init__(self):
        self.direction = ['minimize']
        
    def _objective(self, trail):

        gc.collect()
        torch.cuda.empty_cache()
        
        params_trail = self._params(trail)

        if params_trail['LR_PATIENCE'] > params_trail['PATIENCE']:
            raise optuna.TrialPruned()

        try:
            loss = train_clr(params=params_trail, save_model=opt_params['SAVE_MODELS'], save_path=c.optuna_resnet_path(trail.number), experiment_tracking=True)
            if opt_params['SAVE_REPRESENTATIONS']:
                write_representation(c.optuna_resnet_path(trail.number), c.optuna_representation_path, params = params_trail)
            return loss

        except torch.cuda.OutOfMemoryError:

            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
            

class ParameterOptimizationCLR_All(ParameterOptimizationCLR):
    
    def __init__(self):
        self.direction = ['minimize']
    
    def _params(self, trail):

        AUGMENTATION_PARAMS = {'ROTATION': 15,
                               'TRANSLATE': trail.suggest_float('TRANSLATE', 0.05, 0.5),
                               'SCALE': trail.suggest_float('SCALE', 1.2, 2.5),
                               'FLIP': True,
                               'GAUSSIAN_BLUR_SIGMA': [0.001, trail.suggest_float('GAUSSIAN_BLUR_SIGMA', 0.01, 10)],
                               'NOISE_STD': [0.01, trail.suggest_float('NOISE_STD', 0.02, 0.1)]}

        params_trail = {'BATCH_SIZE': trail.suggest_int('BATCH_SIZE', 16, 128, log=True),
                        'RESNET_DROPOUT':  trail.suggest_float('RESNET_DROPOUT', 0.1, 0.5),
                        'RESNET_REPRESENTATION_DIM': trail.suggest_categorical('RESNET_REPRESENTATION_DIM', [64, 128, 256,512]),
                        'RESNET_REPRESENTATION_DEPTH': trail.suggest_int('RESNET_REPRESENTATION_DEPTH', 1, 3),
                        'RESNET_PROJECTION_DIM': trail.suggest_categorical('RESNET_PROJECTION_DIM', [64, 128, 256,512]),
                        'RESNET_PROJECTION_DEPTH': trail.suggest_int('RESNET_PROJECTION_DEPTH', 1, 3),
                        'NNCLR_QUEUE_SIZE': trail.suggest_int('NNCLR_QUEUE_SIZE', 64, 4096),
                        'nce_temperature': trail.suggest_float('nce_temperature', 0.01, 0.1),
                        'LEARNING_RATE': trail.suggest_float('LEARNING_RATE', 0.0001, 0.005, log=True),
                        'LR_PATIENCE': trail.suggest_int('LR_PATIENCE', 3, 20),
                        'LR_DECAY': trail.suggest_float('LR_DECAY', 0.2, 0.8),
                        'L2_DECAY':  trail.suggest_float('L2_DECAY',  0.00001, 0.001, log=True),
                        'PATIENCE': 30,
                        'AUGMENTATION_PARAMS': AUGMENTATION_PARAMS}

        return params_trail

            
if __name__ == "__main__":
    
    module = __import__("scripts.model.params_opt", globals(), locals(), opt_params['STUDY_OBJECT'])
    opt = getattr(module, opt_params['STUDY_OBJECT'])()
    opt.run()
