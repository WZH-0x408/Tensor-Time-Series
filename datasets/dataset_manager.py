import importlib
import os

class DatasetAPIBase:
    def __init__(self) -> None:
        pass
    def len(self):
        return 0
    def get_data_by_idx(self, index)->dict:
        return index

class DatasetManager:
    def __init__(self) -> None:
        self.manager_dir = os.path.dirname(__file__)
        
    def get_dataset_api(self, dataset_name:str)->DatasetAPIBase:
        module_name = f"datasets.{dataset_name}.dataset_api"
        module = importlib.import_module(module_name)
        ds_api_class = getattr(module, 'DatasetAPI')
        return ds_api_class


if __name__ == '__main__':
    import argparse
    # cmd to add dataset
    pass