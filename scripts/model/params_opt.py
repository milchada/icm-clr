'''

Script to optimize the model using optuna

'''

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

import os

from scripts.model.train_simclr import train_simclr
#from scripts.model.train_cinn import train_cinn
from scripts.util.make_dir import make_dir
import config as c

class ParameterOptimization:
    
    def __init__(self):
        pass
    
    def _objective(self, trail):
        pass
        
    def _create_study(self):
        optuna_storage = JournalStorage(JournalFileStorage(c.optuna_storage))
        self.study = optuna.create_study(directions=self.direction, load_if_exists=True, storage=optuna_storage, study_name=self.study_name)
        
    def run(self, timeout=24*3600):
        self._create_study()
        self.study.optimize(self._objective, timeout=timeout, gc_after_trial=True)
        print(self.study.best_trial.value)
    
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

        params_trail = {'BATCH_SIZE': trail.suggest_int('BATCH_SIZE', 16, 128, log=True),
                        'MAX_NUM_BATCHES_PER_EPOCH': trail.suggest_int('BATCH_SIZE', 16, 128, log=True),
                        'RESNET_DEPTH': trail.suggest_categorical('RESNET_DEPTH', [10, 16]), #6n+4
                        'RESNET_WIDTH': trail.suggest_int('RESNET_WIDTH', 1, 2),
                        'RESNET_DROPOUT':  trail.suggest_float('RESNET_DROPOUT', 0.1, 0.5),
                        'RESNET_REPRESENTATION_DIM': trail.suggest_categorical('RESNET_REPRESENTATION_DIM', [64, 128, 256]),
                        'RESNET_REPRESENTATION_DEPTH': trail.suggest_int('RESNET_REPRESENTATION_DEPTH', 1, 3),
                        'RESNET_PROJECTION_DIM': trail.suggest_categorical('RESNET_PROJECTION_DIM', [64, 128, 256]),
                        'RESNET_PROJECTION_DEPTH': trail.suggest_int('RESNET_PROJECTION_DEPTH', 1, 3)}

        loss = train_simclr(params=params_trail, save_model=False, experiment_tracking=True)

        return loss

class ParameterOptimizationNNCLR(ParameterOptimization):
    
    def __init__(self):
        self.study_name = 'nnclr'
        self.plot_path = c.plots_path + "optuna/nnclr/"
        self.direction = ['minimize']
    
    def _objective(self, trail):

        params_trail = {'BATCH_SIZE': trail.suggest_int('BATCH_SIZE', 16, 128, log=True),
                        'RESNET_DEPTH': trail.suggest_categorical('RESNET_DEPTH', [10, 16]), #6n+4
                        'RESNET_WIDTH': trail.suggest_int('RESNET_WIDTH', 1, 2),
                        'RESNET_DROPOUT':  trail.suggest_float('RESNET_DROPOUT', 0.1, 0.5),
                        'RESNET_REPRESENTATION_DIM': trail.suggest_categorical('RESNET_REPRESENTATION_DIM', [64, 128, 256]),
                        'RESNET_REPRESENTATION_DEPTH': trail.suggest_int('RESNET_REPRESENTATION_DEPTH', 1, 3),
                        'RESNET_PROJECTION_DIM': trail.suggest_categorical('RESNET_PROJECTION_DIM', [64, 128, 256]),
                        'RESNET_PROJECTION_DEPTH': trail.suggest_int('RESNET_PROJECTION_DEPTH', 1, 3),
                        'NNCLR_QUEUE_SIZE': trail.suggest_int('NNCLR_QUEUE_SIZE', 64, 1028)}

        loss = train_simclr(params=params_trail, save_model=False, experiment_tracking=True)

        return loss
'''
class ParameterOptimizationCINN(ParameterOptimization):
    
    def __init__(self):
        self.study_name = 'cinn'
        self.plot_path = c.plots_path + "optuna/cinn/"
        self.direction = ['minimize']
    
    def _objective(self, trail):

        batch_norm = trail.suggest_categorical('BATCH_NORM', [True, False])
        dropout = trail.suggest_float('DROPOUT', 0., 0.4, step=0.1)

        params_trail = {'BATCH_SIZE': trail.suggest_int('BATCH_SIZE', 256, 4096, log=True),
                        'LEARNING_RATE': trail.suggest_float('LEARNING_RATE', 0.0001, 0.005, log=True),
                        'LR_DECAY':  trail.suggest_float('LR_DECAY', 0.1, 0.9),
                        'NUM_COUPLING_LAYERS': trail.suggest_int('NUM_COUPLING_LAYERS', 6, 14),
                        'NUM_COND_NODES': trail.suggest_categorical('NUM_COND_NODES', [64, 128, 256]),
                        'NUM_HIDDEN_NODES': trail.suggest_categorical('NUM_HIDDEN_NODES', [64, 128, 256]),
                        'NUM_HIDDEN_LAYERS': trail.suggest_int('NUM_HIDDEN_LAYERS', 1, 3),
                        'NUM_HIDDEN_NODES_COND': trail.suggest_categorical('NUM_HIDDEN_NODES_COND', [64, 128, 256]),
                        'NUM_HIDDEN_LAYERS_COND': trail.suggest_int('NUM_HIDDEN_LAYERS_COND', 1, 3),
                        'L2_DECAY': trail.suggest_float('L2_DECAY', 0.00001, 0.0001),
                        'NOISE': trail.suggest_float('NOISE', 0.001, 0.05, log=True),
                        'BATCH_NORM': batch_norm,
                        'BATCH_NORM_COND': batch_norm,
                        'DROPOUT': dropout,
                        'DROPOUT_COND': dropout}

        nll, fmmd, bmmd, z_std, z_mean = train_cinn(params=params_trail, save_model=False, experiment_tracking=False)

        return nll
'''

if __name__ == "__main__":

    opt = ParameterOptimizationNNCLR()
    opt.run()
    opt.plot()
