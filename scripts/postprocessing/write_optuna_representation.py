import optuna
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileSymlinkLock
from scripts.util.logging import logger
import config as c
from scripts.postprocessing.write_representation import write_representation

import yaml

#Load the default params from the parameter file
opt_params = yaml.safe_load(open('params.yaml'))['params_opt']

#Load study
optuna_storage_lock = JournalFileSymlinkLock(c.optuna_storage)
optuna_storage = JournalStorage(JournalFileStorage(c.optuna_storage, lock_obj=optuna_storage_lock))
study = optuna.load_study(study_name=opt_params['STUDY_NAME'], storage=optuna_storage)

trials = study.get_trials()

logger.info(str(len(trials)) + ' trials found.')

for trial in trials:
    if not trial.state.COMPLETE:
        logger.info(str(trial.number) + ' not completed. Skipping...')
        continue
    
    logger.info(str(trial.number) + ' is loaded and representations calculated.')
    write_representation(c.optuna_resnet_path(trial.number), c.optuna_representation_path(trial.number), params = trial.params)
    
    

