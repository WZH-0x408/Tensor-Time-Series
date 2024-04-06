import argparse
import shutil
import os

if __name__ == '__main__':
    # build args parser
    parser = argparse.ArgumentParser(
            description='A stript to add a new Model Integration'
        )
    parser.add_argument(
            '--name', type=str, required=True,
            help='The name of the new Model Integration.'
        )

    # parse
    args = parser.parse_args()

    # get name
    model_int_name = args.name
    
    # check name
    models_dir = os.path.dirname(__file__)
    model_list = os.listdir(models_dir)
    if model_int_name in model_list:
        raise ValueError(f"The {model_int_name} ModelInt already exists...")
    
    # add
    template_dir = os.path.join(models_dir, 'template')
    target_dir = os.path.join(models_dir, f"{model_int_name}")

    # copy and paste
    shutil.copytree(template_dir, target_dir)
    
