import os
import yaml
import argparse
import random
import numpy as np
import torch
from task.Trainer import Trainer
from task.Tester import Tester
from task.Logger import Logger

class Task:
    def __init__(self, root_dir:str, configs:dict) -> None:
        self.root_dir = root_dir
        self.configs = configs
        self.seed_everything()
        self.ensure_output_dir()
        self.logger = Logger()
        Logger.set_backend(self.out_dir, configs=configs, backend=configs['Task']['logger'])

    def seed_everything(self):
        seed = self.configs['Task']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)        # single GPU
            # torch.cuda.manual_seed_all(seed)  # multi GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def ensure_output_dir(self):
        project = self.configs['Task']['project']
        runID = self.configs['Task']['runID']
        mode = self.configs['Task']['mode']
        out_dir = os.path.join(self.root_dir, project, f"{mode}-{runID}")
        if os.path.exists(out_dir):
            num = 1
            while os.path.exists(f"{out_dir}_{num}"):
                num += 1
            out_dir = f"{out_dir}_{num}"
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.configs['Task']['out_dir'] = out_dir
        with open(os.path.join(out_dir, 'meta.yaml'), 'w') as file:
            yaml.dump(self.configs, file)

    def close(self):
        Logger.close()

    def run(self):
        mode = self.configs['Task']['mode']
        if mode == 'train':
            trainer = Trainer(self.configs, self.logger)
            trainer.train()
        if mode == 'test':
            tester = Tester(self.configs, self.logger)
            result_info = tester.test()
            print(result_info)
        self.close()

if __name__ == '__main__':
    
    # build an args parser
    parser = argparse.ArgumentParser(
        description='Tensor-Time-Series Libs'
    )
    parser.add_argument(
        '-F', '--file', type=str, required=True,
        help='The path of yaml config file.'
    )
    parser.add_argument(
        '--out', type=str, default='./output',
        help='the path of output.'
    )
    # parse
    args = parser.parse_args()

    # get file path
    configs = yaml.safe_load(open(args.file))
    root_dir = args.out
    # Create a Task
    task = Task(root_dir, configs)
    task.run()

        
    