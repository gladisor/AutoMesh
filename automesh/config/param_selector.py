## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import yaml
from typing import Tuple, Dict


from optuna import Trial
from optuna.trial import FixedTrial


class ParamSelector:
    def __init__(self, trial: Trial):
        self.trial = trial
        #self.select_params('conv_layer')
        
        #params=self.get_basic_params()
        
        
    def select_params(self, config_key: str) -> Tuple[object, Dict]:
        path=os.path.join('automesh/config/' + config_key + '.yml')
        with open(path, 'r') as stream:
            try:
                parsed_yaml=yaml.full_load(stream)
               
            except yaml.YAMLError as exc:
                pass
        obj_names = list(parsed_yaml.keys())
      
        obj_name = self.trial.suggest_categorical(config_key, obj_names)
        
        obj = parsed_yaml[obj_name]['obj']  
        params={}
        
        if 'params' in parsed_yaml[obj_name]: 
            for arg in parsed_yaml[obj_name]['params']:
                value=parsed_yaml[obj_name]['params'][arg]
                if type(value) == list:
                    params[arg] = self.trial.suggest_categorical(arg, value)
                elif type(value) == tuple:
                    if type(value[0]) == int:
                        params[arg] = self.trial.suggest_int(arg, value[0], value[1])
                    elif type(value[0])== float:
                        params[arg] = self.trial.suggest_float(arg, value[0], value[1])
          
        return (obj, params)
        
    
        #suggesting base parameters: currently only hidden_channels, num_layers, lr
    def get_basic_params(self,config_key: str) ->  Dict:
    
        path=os.path.join('automesh/config/'+ config_key +'.yml')
        with open(path, 'r') as stream:
            try:
                parsed_yaml=yaml.full_load(stream)
            except:
                pass
        
        params_names = list(parsed_yaml.keys())       
        params={}
        
        for param_name in params_names:
            value=parsed_yaml[param_name]            
            if type(value) == tuple:
                if type(value[0]) == int:
                    params[param_name] = self.trial.suggest_int(param_name, value[0], value[1])
                elif type(value[0])== float:
                    params[param_name] = self.trial.suggest_float(param_name, value[0], value[1])
            elif type(value) == list:
                params[param_name] = self.trial.suggest_categorical(param_name, value)
            elif (type(value)== int) or (type(value)==float):
                params[param_name]=value
                    
        return params
 