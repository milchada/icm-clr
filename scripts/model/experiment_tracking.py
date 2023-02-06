import yaml
from pandas.io.json._normalize import nested_to_record

class ExperimentTracking(object):
    
    def __init__(self):
        self.set_up()
        
    def set_up(self):
        pass
        
    def log_metric(self, x):
        pass
    
class NeptuneExperimentTracking(ExperimentTracking):
    
    def __init__(self, params, tags=[]):
        self._params = params
        self._tags = tags
        self.set_up()
    
    def set_up(self):
        
        default_params = yaml.safe_load(open('params.yaml'))
        neptune_params = default_params['neptune']
        params = dict(nested_to_record(default_params), **params)
    
        try:
            import neptune
            project = neptune.init(project_qualified_name=neptune_params["project_name"],
                                   api_token=neptune_params["api"])
            
            #Use name of branch as experiment name
            from git import Repo
            experiment_name = Repo('./').active_branch.name
            
            #Create neptune experiment and save all parameters in the parameter file
            self._experiment = project.create_experiment(name=experiment_name, 
                                                         params=params,
                                                         tags=neptune_params["tags"] + self._tags)
            
        except:
            print("WARNING: Neptune init failed. Experiment tracking via neptune disabled!")
            self._experiment = None
            
    def log_metric(self, x):
        if self._experiment is not None:
            for key, value in zip(list(x.keys()), list(x.values())):
                self._experiment.log_metric(key, value)
                
class VoidExperimentTracking(ExperimentTracking):
    pass
        
class JsonExperimentTracking(ExperimentTracking):
    def set_up(self):
        pass
        
    def log_metric(x):
        pass
