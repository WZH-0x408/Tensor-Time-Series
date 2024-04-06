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
    '''
    command line tool: Add a new dataset
    '''
    import argparse
    import shutil
    parser = argparse.ArgumentParser(
            description='A stript to add a new Dataset'
        )
    parser.add_argument(
            '--name', type=str, required=True,
            help='The name of the new Dataset.'
        )

    # parse
    args = parser.parse_args()

    # get name
    dataset_name = args.name
    
    # check name
    models_dir = os.path.dirname(__file__)
    model_list = os.listdir(models_dir)
    if dataset_name in model_list:
        raise ValueError(f"The {dataset_name} Dataset already exists...")
    
    # add
    template_dir = os.path.join(models_dir, 'template_dataset')
    target_dir = os.path.join(models_dir, f"{dataset_name}")

    # copy and paste
    shutil.copytree(template_dir, target_dir)
    
