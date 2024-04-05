import argparse
import shutil
import os

if __name__ == '__main__':
    # build args parser
    parser = argparse.ArgumentParser(
            description='A stript to add a new ModelPkt'
        )
    parser.add_argument(
            '--name', type=str, required=True,
            help='The name of the new ModelPkt.'
        )

    # parse
    args = parser.parse_args()

    # get name
    model_pkt_name = args.name
    
    # check name
    models_dir = os.path.dirname(__file__)
    model_list = os.listdir(models_dir)
    if model_pkt_name in model_list:
        raise ValueError(f"The {model_pkt_name} ModelPkt already exists...")
    
    # add
    template_dir = os.path.join(models_dir, 'template')
    target_dir = os.path.join(models_dir, f"{model_pkt_name}")

    # copy and paste
    shutil.copytree(template_dir, target_dir)
    
