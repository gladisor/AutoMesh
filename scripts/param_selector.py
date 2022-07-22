## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from typing import Tuple, Dict


from optuna import Trial


class ParamSelector:
    def __init__(self, trial: Trial):
        self.trial = trial
       # self.select_params('layers.yml')
       
    def select_params(self, config_key: str) -> Tuple[object, Dict]:
        path=os.path.join('config/' + config_key + '.yml')
        with open(path, 'r') as stream:
            try:
                parsed_yaml=yaml.full_load(stream)
                #print(parsed_yaml)
            except yaml.YAMLError as exc:
                pass
        
        obj_names = list(parsed_yaml.keys())
        #print('obj names', obj_names)
        #conv layer ändern aus name ziehen 
        obj_name = self.trial.suggest_categorical(config_key, obj_names)
        
        obj = parsed_yaml[obj_name]['obj']  
        params={}
        
        # try:
        #     flag=parsed_yaml[obj_name]['params']
        # except Exception:
        #     #print("kein Param hier" )
        #     pass
        # if flag!=0:
        #     flag=0  
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
        #print(obj_name)
        #print(params)
                    
                    
        return (obj, params)
        


#if __name__ == '__main__':
    #study = create_study()
   # study.optimize(ParameterSelector, n_trials = 20)
    #trial = FixedTrial({'hidden_channels': 832, 'conv_layer': 'GATConv'})
    #ps = ParameterSelector(trial)
    # ps.select_params('layers.yml')
 