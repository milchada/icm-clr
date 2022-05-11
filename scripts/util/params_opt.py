'''

Script to optimize the model using optuna

STILL UNDER CONSTRUCTION

'''

import optuna
import joblib

import os

from scripts.model.train_simclr import train_simclr
from scripts.model.train_cinn import train_cinn
from scripts.util.make_dir import make_dir
import config as c

class ParameterOptimization:
    
    def __init__(self):
        pass
    
    def _objective(self, trail):
        pass
       
    def _checkpoint_callback(self, study, trial):
        make_dir(self.study_path)
        joblib.dump(study, self.study_path)
        
    def _create_study(self, continue_study):
        if os.path.exists(self.study_path) and continue_study:
            self.study = joblib.load(self.study_path)
        else:
            self.study = optuna.create_study(directions=self.direction)
        
    def run(self, timeout=22*3600, continue_study=True):
        self._create_study(continue_study)
        self.study.optimize(self._objective, timeout=timeout, gc_after_trial=True, callbacks=(self._checkpoint_callback))
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
        self.plot_path = c.plots_path + "optuna/resnet/"
        self.study_path = c.metrics_path + "optuna_resnet.pkl"
        self.direction = ['minimize', 'maximize']
    
    def _objective(self, trail):

        params_trail = {'BATCH_SIZE': trail.suggest_int('BATCH_SIZE', 32, 256, log=True),
                        'LEARNING_RATE': trail.suggest_float('LEARNING_RATE', 0.0001, 0.001, log=True),
                        'WEIGHT_DECAY':  trail.suggest_float('WEIGHT_DECAY', 0.00001, 0.001, log=True),
                        'RESNET_DEPTH': trail.suggest_categorical('RESNET_DEPTH', [10, 16]), #6n+4
                        'RESNET_WIDTH': trail.suggest_int('RESNET_WIDTH', 1, 2),
                        'RESNET_DROPOUT':  trail.suggest_float('RESNET_DROPOUT', 0.1, 0.5),
                        'RESNET_REPRESENTATION_DIM': trail.suggest_categorical('RESNET_REPRESENTATION_DIM', [64, 128, 256]),
                        'RESNET_PROJECTION_DIM': trail.suggest_categorical('RESNET_PROJECTION_DIM', [64, 128, 256]),
                        'RESNET_PROJECTION_DEPTH': trail.suggest_int('RESNET_PROJECTION_DEPTH', 1, 2)}

        loss, top1, top5 = train_simclr(params=params_trail, save_model=False, experiment_tracking=True)

        return loss, top1

class ParameterOptimizationCINN(ParameterOptimization):
    
    def __init__(self):
        self.plot_path = c.plots_path + "optuna/cinn/"
        self.study_path = c.metrics_path + "optuna_cinn.pkl"
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


if __name__ == "__main__":
    
    opt = ParameterOptimizationSimCLR()
    opt.run()
    opt.plot()