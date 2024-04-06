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

    
# if __name__ == '__main__':
#     '''
#     command line tool: add new model_int
#     '''
#     import shutil
#     import argparse
#     # build args parser
#     parser = argparse.ArgumentParser(
#             description='A stript to add a new ModelPkt'
#         )
#     parser.add_argument(
#             '--name', type=str, required=True,
#             help='The name of the new ModelPkt.'
#         )

#     # parse
#     args = parser.parse_args()

#     # get name
#     model_pkt_name = args.name
    
#     # check name
#     models_dir = os.path.dirname(__file__)
#     model_list = os.listdir(models_dir)
#     if model_pkt_name in model_list:
#         raise ValueError(f"The {model_pkt_name} ModelPkt already exists...")
    
#     # add
#     template_dir = os.path.join(models_dir, 'template')
#     target_dir = os.path.join(models_dir, f"{model_pkt_name}")

#     # copy and paste
#     shutil.copytree(template_dir, target_dir)
    


    