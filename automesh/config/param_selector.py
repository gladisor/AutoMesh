## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import yaml
from typing import Tuple, Dict, List
from optuna import Trial

#class to select categorical and numerical hyperparameters for optuna study  
class Selector:
    def __init__(self, trial: Trial, spec: List[str]):
        self.trial = trial
        self.spec = spec
        
        for file in spec:
            setattr(self, file, self.select(file))
        
 
    def read_yml(self, name: str) -> Dict:
        path=os.path.join('automesh/config/'+ name +'.yml')
        with open(path, 'r') as stream:
            parsed_yml = yaml.full_load(stream)
        
        return parsed_yml
    #uses optuna.suggest* to suggest values in different categories in sweep
    def suggest(self, key, value):
   
        if type(value) == tuple:
            if type(value[0]) == int:
                value = self.trial.suggest_int(key, value[0], value[1])
            elif type(value[0])== float:
                value = self.trial.suggest_float(key, value[0], value[1])
        elif type(value) == list:
            value = self.trial.suggest_categorical(key, value)
        else:
            pass
        
        return value
    
    
    #selects recursively parameters from yml files
    def select(self, config_key) -> Tuple:
        
        parsed_yml = self.read_yml(config_key)
        obj_names = list(parsed_yml.keys())
        obj_name = self.trial.suggest_categorical(config_key, obj_names)
        
        # if file simply contains a list of categorical options simply return selection
        if 'params' not in parsed_yml[obj_name]:
            return (parsed_yml[obj_name]['obj'], {})
        else:
            obj_params = parsed_yml[obj_name]['params']
        
        d = {}
        #3 options for param location: 1)in seperate yml file 2)in "basic" yml file
        #3)in the current yml file just as argument to the key
        for key in obj_params:
            if parsed_yml[obj_name]['params'][key] == 'not_basic':
                
                obj, kwargs = self.select(key)                
                d={**d, key+'_kwargs': kwargs, key: obj}
      
            elif parsed_yml[obj_name]['params'][key] == 'basic':
                
                basic = self.read_yml('basic')
                d.update({
                    key: self.suggest(key, basic[key])
                    })
                
            else:
                d.update({
                    key: self.suggest(key, parsed_yml[obj_name]['params'][key])
                    })
                
        return (parsed_yml[obj_name]['obj'], d)
        
    #creates dict in a way that heatmapregressor can take it as input
    #every independent toplevel parameter has an obj selection and selected kwargs    
    def params(self) -> Dict:
        d = {}
        
        for file in self.spec:
            obj, obj_kwargs = getattr(self, file)
            d[file] = obj     
            d[file + '_kwargs'] = obj_kwargs
            
        return d
    

          


    
        
    
    
    