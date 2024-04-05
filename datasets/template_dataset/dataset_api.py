from torch.utils.data import Dataset
import numpy as np
import os

from ..dataset_manager import DatasetAPIBase
'''
read or generate data
'''
def gen_template_dataset(N = 3000):
    data = np.zeros((N, 2), dtype=int)
    label = np.zeros(N,dtype=int)
    for i in range(N):
        data[i,0] = i%2==0
        data[i,1] = i%3==0
        label[i] = i%6==0
    return data, label
'''
Rewrite DatasetAPI to fit dataset
'''
class DatasetAPI(DatasetAPIBase):
    def __init__(self, N=1000):
        self.data, self.label = gen_template_dataset(N)
    def len(self):
        return len(self.data)
    def get_data_by_idx(self, index)->dict:
        item = {
            'data': self.data[index],
            'label': self.label[index]
        }
        return item