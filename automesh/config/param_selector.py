## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import yaml
from typing import Tuple, Dict, List


from optuna import Trial
from optuna.trial import FixedTrial
from automesh.models.architectures import ParamGCN

    
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
        
        
    def params(self) -> Dict:
        d = {}
        
        for file in self.spec:
            obj, obj_kwargs = getattr(self, file)
            d[file] = obj     
            d[file + '_kwargs'] = obj_kwargs
            
        return d
    

# class ParamSelector:
#     def __init__(self, trial: Trial):
#         self.trial = trial
  
    
# #select trials from config .yml files. different initial categorical choices of
# #the initial suggest_categorical function affect the chooseable hyperparameters
# #down the line   
#     def select_params(self, config_key: str) -> Tuple[object, Dict]:
#         path=os.path.join('automesh/config/' + config_key + '.yml')
#         with open(path, 'r') as stream:
#             try:
#                 parsed_yaml=yaml.full_load(stream)
               
#             except yaml.YAMLError as exc:
#                 pass
#         obj_names = list(parsed_yaml.keys())
      
#         obj_name = self.trial.suggest_categorical(config_key, obj_names)
        
#         obj = parsed_yaml[obj_name]['obj']  
#         params={}
        
#         if 'params' in parsed_yaml[obj_name]: 
#             for arg in parsed_yaml[obj_name]['params']:
#                 value=parsed_yaml[obj_name]['params'][arg]
#                 if type(value) == list:
#                     params[arg] = self.trial.suggest_categorical(arg, value)
#                 elif type(value) == tuple:
#                     if type(value[0]) == int:
#                         params[arg] = self.trial.suggest_int(arg, value[0], value[1])
#                     elif type(value[0])== float:
#                         params[arg] = self.trial.suggest_float(arg, value[0], value[1])
          
#         return (obj, params)
        
    
#         #suggesting base parameters: currently only hidden_channels, num_layers, lr
#     def get_basic_params(self,config_key: str) ->  Dict:
    
#         path=os.path.join('automesh/config/'+ config_key +'.yml')
#         with open(path, 'r') as stream:
#             try:
#                 parsed_yaml=yaml.full_load(stream)
#             except:
#                 pass
        
#         params_names = list(parsed_yaml.keys())       
#         params={}
        
#         for param_name in params_names:
#             value=parsed_yaml[param_name]            
#             if type(value) == tuple:
#                 if type(value[0]) == int:
#                     params[param_name] = self.trial.suggest_int(param_name, value[0], value[1])
#                 elif type(value[0])== float:
#                     params[param_name] = self.trial.suggest_float(param_name, value[0], value[1])
#             elif type(value) == list:
#                 params[param_name] = self.trial.suggest_categorical(param_name, value)
#             elif (type(value)== int) or (type(value)==float):
#                 params[param_name]=value
                    
#         return params
    
    
#     #gets all parameters and passes them to the right place 
#     def param_passing(self):
        
#         basic_params=self.get_basic_params('basic')
#         conv_layer, conv_layer_kwargs = self.select_params('conv_layer')
#         act, act_kwargs = self.select_params('act')
#         norm, norm_kwargs = self.select_params('norm')
#         loss_func, loss_func_kwargs = self.select_params('loss_func')
#         opt, opt_kwargs = self.select_params('opt')
    
        
#         all_params = {
#              'base': ParamGCN, ##hard coded but might be parameterized later
#              'base_kwargs': {
#                  'conv_layer': conv_layer,
#                  'conv_layer_kwargs': conv_layer_kwargs,
#                  'in_channels': basic_params['in_channels'],
#                  'hidden_channels': basic_params['hidden_channels'],
#                  'num_layers': basic_params['num_layers'],
#                  'out_channels': basic_params['out_channels'],
#                  'act': act,
#                  'act_kwargs': act_kwargs,
#                  'norm': norm(basic_params['hidden_channels'])
#              },
#              'loss_func': loss_func,
#              'loss_func_kwargs': loss_func_kwargs,
#              'opt': opt,
#              'opt_kwargs': {'lr' : basic_params['lr'], **opt_kwargs}
#          }
         
         
#         return all_params
    

        
        
    # def select(self,config_key: str, obj_name='', state=''):
    #     param_keys=[]
    #     path=os.path.join('automesh/config/'+ config_key +'.yml')
    #     with open(path, 'r') as stream:
    #         parsed_yml=yaml.full_load(stream)
    #     #get toplevel categorical, if file is opened the first time: suggest
    #     # if it is opened with saved_state choose saved_state obj_name
    #     if obj_name=='':
    #         obj_names = list(parsed_yml.keys())
    #         obj_name = self.trial.suggest_categorical(config_key, obj_names)
            
    #     #write all params to list which is evaluated in the next step
    #     #if there is a saved state, ignore all already evaluated parameters
    #     for key in parsed_yml[obj_name]['params']:
    #         if state=='':
    #             param_keys.append(key)
    #         elif state==key:
    #             state=''
    #     #check what type key is: directly specified, config file name, basic param
    #     for key in param_keys:
    #         if parsed_yml[obj_name]['params'][key]!=None:
    #             self.suggest(key, parsed_yml[obj_name]['params'][key])
    #         elif os.path.exists('automesh/config/'+ key+'.yml'):
    #             #elf.hierarchy_level+=1
    #             self.saved_state.extend((config_key, obj_name, key))
    #             #safe state of level 0 param list
    #             self.select(key)
    #         else: 
    #             path=os.path.join('automesh/config/'+ 'basic' +'.yml')
    #             with open(path, 'r') as stream:
    #                 parsed_basic_yml=yaml.full_load(stream)
    #             self.suggest(key, parsed_basic_yml[key])    
                
                
    #     #check if all parameters in all hierarchical are evaluated
    #     if self.saved_state!=[]:
    #         self.select(self.saved_state[-3], 
    #                     self.saved_state[-2], 
    #                     self.saved_state[-1])
    #         del self.saved_state[-3:]               
                
               
                


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    