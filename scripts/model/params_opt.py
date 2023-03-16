'''

Script to optimize the model using optuna

'''

import optuna
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileSymlinkLock

import os
import torch
import gc


from scripts.model.train_simclr import train_clr
#from scripts.model.train_cinn import train_cinn
from scripts.util.make_dir import make_dir
import config as c

class ParameterOptimization:
    
    def __init__(self):
        pass
    
    def _objective(self, trail):
        pass
        
    def _create_study(self):
        optuna_storage_lock = JournalFileSymlinkLock(c.optuna_storage)
        optuna_storage = JournalStorage(JournalFileStorage(c.optuna_storage, lock_obj=optuna_storage_lock))
        self.study = optuna.create_study(directions=self.direction, load_if_exists=True, storage=optuna_storage, study_name=self.study_name)
        
    def run(self):
        self._create_study()
        self.study.optimize(self._objective, gc_after_trial=True, catch=(torch.cuda.OutOfMemoryError))
    
    def plot(self):
        '''Save study plots'''
        
        make_dir(self.plot_path)

        fig = optuna.visualization.plot_contour(self.study)
        fig.write_image(self.plot_path + "contour.pdf")

        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.write_image(self.plot_path + "optimization_history.pdf")

        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.write_image(self.plot_path + "parallel_coordinate.pdf")

        fig = optuna.visualization.plot_slice(self.study)
        fig.write_image(self.plot_path + "slice.pdf")

        fig = optuna.visualization.plot_param_importances(self.study)
        fig.write_image(self.plot_path + "param_importances.pdf")
                
    
class ParameterOptimizationSimCLR(ParameterOptimization):
    
    def __init__(self):
        self.study_name = 'simclr'
        self.plot_path = c.plots_path + "optuna/simclr/"
        self.direction = ['minimize']
    
    def _objective(self, trail):
        gc.collect()
        torch.cuda.empty_cache()

        params_trail = {'BATCH_SIZE': trail.suggest_int('BATCH_SIZE', 16, 128, log=True),
                        'RESNET_DEPTH': trail.suggest_categorical('RESNET_DEPTH', [10, 16]), #6n+4
                        'RESNET_WIDTH': trail.suggest_int('RESNET_WIDTH', 1, 2),
                        'RESNET_DROPOUT':  trail.suggest_float('RESNET_DROPOUT', 0.1, 0.5),
                        'RESNET_REPRESENTATION_DIM': trail.suggest_categorical('RESNET_REPRESENTATION_DIM', [64, 128, 256]),
                        'RESNET_REPRESENTATION_DEPTH': trail.suggest_int('RESNET_REPRESENTATION_DEPTH', 1, 3),
                        'RESNET_PROJECTION_DIM': trail.suggest_categorical('RESNET_PROJECTION_DIM', [64, 128, 256]),
                        'RESNET_PROJECTION_DEPTH': trail.suggest_int('RESNET_PROJECTION_DEPTH', 1, 3)}

        try:

            loss = train_clr(params=params_trail, save_model=False, experiment_tracking=True)
            return loss

        except torch.cuda.OutOfMemoryError:

            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()


class ParameterOptimizationNNCLR(ParameterOptimization):
    
    def __init__(self):
        self.study_name = 'nnclr'
        self.plot_path = c.plots_path + "optuna/nnclr/"
        self.direction = ['minimize']
    
    def _objective(self, trail):

        gc.collect()
        torch.cuda.empty_cache()

        params_trail = {'BATCH_SIZE': trail.suggest_int('BATCH_SIZE', 16, 128, log=True),
                        'RESNET_DEPTH': trail.suggest_categorical('RESNET_DEPTH', [10, 16]), #6n+4
                        'RESNET_WIDTH': trail.suggest_int('RESNET_WIDTH', 1, 2),
                        'RESNET_DROPOUT':  trail.suggest_float('RESNET_DROPOUT', 0.1, 0.5),
                        'RESNET_REPRESENTATION_DIM': trail.suggest_categorical('RESNET_REPRESENTATION_DIM', [64, 128, 256,512]),
                        'RESNET_REPRESENTATION_DEPTH': trail.suggest_int('RESNET_REPRESENTATION_DEPTH', 1, 3),
                        'RESNET_PROJECTION_DIM': trail.suggest_categorical('RESNET_PROJECTION_DIM', [64, 128, 256,512]),
                        'RESNET_PROJECTION_DEPTH': trail.suggest_int('RESNET_PROJECTION_DEPTH', 1, 3),
                        'NNCLR_QUEUE_SIZE': trail.suggest_int('NNCLR_QUEUE_SIZE', 64, 2048)}

        try:

            loss = train_clr(params=params_trail, save_model=False, experiment_tracking=True)
            return loss

        except torch.cuda.OutOfMemoryError:

            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()


class ParameterOptimizationCLR(ParameterOptimization):
    
    def __init__(self):
        self.study_name = 'nnclr_v2'
        self.plot_path = c.plots_path + "optuna/nnclr_v2/"
        self.direction = ['minimize']
    
    def _objective(self, trail):

        gc.collect()
        torch.cuda.empty_cache()

        AUGMENTATION_PARAMS = {'ROTATION': 15,
                               'TRANSLATE': trail.suggest_float('TRANSLATE', 0.05, 0.5),
                               'SCALE': trail.suggest_float('SCALE', 1.2, 2.5),
                               'FLIP': True,
                               'GAUSSIAN_BLUR_SIGMA': [0.001, trail.suggest_float('GAUSSIAN_BLUR_SIGMA', 0.01, 10)],
                               'NOISE_STD': [0.01, trail.suggest_float('NOISE_STD', 0.02, 0.1)]}

        params_trail = {'nce_temperature': trail.suggest_float('nce_temperature', 0.01, 0.1),
                        'LEARNING_RATE': trail.suggest_float('LEARNING_RATE', 0.0001, 0.005, log=True),
                        'LR_PATIENCE': trail.suggest_int('LR_PATIENCE', 3, 20),
                        'LR_DECAY': trail.suggest_float('LR_DECAY', 0.2, 0.8),
                        'L2_DECAY':  trail.suggest_float('L2_DECAY',  0.00001, 0.001, log=True),
                        'PATIENCE': trail.suggest_int('PATIENCE', 5, 20),
                        'AUGMENTATION_PARAMS': AUGMENTATION_PARAMS}

        if params_trail['LR_PATIENCE'] > params_trail['PATIENCE']:
            raise optuna.TrialPruned()

        try:

            loss = train_simclr(params=params_trail, save_model=False, experiment_tracking=True)
            return loss

        except torch.cuda.OutOfMemoryError:

            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
            
class ParameterOptimizationCLRv3(ParameterOptimization):
    
    def __init__(self):
        self.study_name = 'nnclr_v3'
        self.plot_path = c.plots_path + "optuna/nnclr_v3/"
        self.direction = ['minimize']
    
    def _objective(self, trail):

        gc.collect()
        torch.cuda.empty_cache()

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

        if params_trail['LR_PATIENCE'] > params_trail['PATIENCE']:
            raise optuna.TrialPruned()

        try:

            loss = train_simclr(params=params_trail, save_model=False, experiment_tracking=True)
            return loss

        except torch.cuda.OutOfMemoryError:

            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

if __name__ == "__main__":
    
    opt = ParameterOptimizationCLRv3()
    opt.run()
