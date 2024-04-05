# import basic pkgs
from ..model_int_base import ModelIntBase
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
# import your model and dataset
from .model import Template_Model
from .dataset import Template_Dataset
# import necessary pkgs
# import xxx

'''
Model Integration:
    - yaml_loader
    - dataset [trian, valid, test]
    - optimizer
    - criterion
    - model
'''
class ModelInt(ModelIntBase):
    def __init__(self, dataset_name):
        self.support_map = {
            'template_dataset': Template_Dataset,
        }
        self.check_support(dataset_name)
        self.ds_name = dataset_name
        # select components of your task
        self.optim_class = optim.Adam
        self.criterion = nn.CrossEntropyLoss
        self.model_class = Template_Model
        self.dataset_class = self.support_map[dataset_name]

    # return an **instance** of Dataset
    def get_dataset(self, mode='train'):
        return self.dataset_class(mode)
    # return a class 
    def get_model_class(self):
        return self.model_class
    # return a class
    def get_optim_class(self):
        return self.optim_class
    # return a class
    def get_criterion_class(self):
        return self.criterion
    
    
        

    