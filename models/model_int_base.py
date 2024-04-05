from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import yaml
import os
'''
Info: The base class of model integration:
Component:
    - yaml_loader
    - dataset [trian, valid, test]
    - optimizer
    - criterion
    - model
'''
class ModelIntBase:
    # supported datset
    def __init__(self, dataset_name):
        self.support_map = {}
        self.check_support(dataset_name)

    def read_yaml_config(self, path=None) -> dict:
        configs = {}
        if path is not None:
            configs = yaml.safe_load(open(path))
        return configs

    def check_support(self, dataset_name):
        if dataset_name not in self.support_map:
            raise ValueError(f"This dataset {dataset_name} is not support now.")

    def get_dataset(self, mode='train')->Dataset:
        pass

    def get_model_class(self)->nn.Module:
        pass

    def get_optim_class(self):
        return self.optim_class

    def get_criterion_class(self):
        pass

    
# '''
# MakeDatasetLoader
# '''
# def MakeDatasetLoader(
#         dataset: Dataset, batch_size:int, 
#         shuffle=False, pin_memory=False
#     ):
#     loader = DataLoader(dataset, 
#                         batch_size=batch_size, 
#                         shuffle=shuffle, 
#                         pin_memory=pin_memory)
#     return loader


    