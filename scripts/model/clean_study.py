import optuna
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileSymlinkLock
import config as c
import yaml
import os

#Load the default params from the parameter file
opt_params = yaml.safe_load(open('params.yaml'))['params_opt']

#Load study
optuna_storage_lock = JournalFileSymlinkLock(c.optuna_storage)
optuna_storage = JournalStorage(JournalFileStorage(c.optuna_storage, lock_obj=optuna_storage_lock))
study = optuna.load_study(study_name=opt_params['STUDY_NAME'], storage=optuna_storage)

directions = study.directions
trials = study.get_trials()

for i, trial in enumerate(trials):
    if trial.state == optuna.trial.TrialState.RUNNING:
        trials[i].state = optuna.trial.TrialState.WAITING

os.remove(c.optuna_storage)

optuna_storage_lock = JournalFileSymlinkLock(c.optuna_storage)
optuna_storage = JournalStorage(JournalFileStorage(c.optuna_storage, lock_obj=optuna_storage_lock))
new_study = optuna.create_study(directions=directions,
                                           load_if_exists=False,
                                           storage=optuna_storage,
                                           study_name=opt_params['STUDY_NAME'])
new_study.add_trials(trials)

