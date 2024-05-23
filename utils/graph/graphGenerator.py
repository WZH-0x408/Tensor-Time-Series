import numpy as np
import pickle
import scipy
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity

import sys, os
pth=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(pth)) #./Tensor-Time-Series
from datasets.dataset import TTS_Dataset


class GraphGenerator:
    def __init__(self, dataset:TTS_Dataset, ratio:float=0.3, max_sample:int=3000) -> None:
        self.dataset = dataset
        self.tensor_shape = self.dataset.get_tensor_shape()
        self.data = self.dataset.data
        time_range = self.data.shape[0]
        sample_n = min(int(ratio*time_range), max_sample)
        indices = np.random.choice(time_range, sample_n, replace=False)
        self.data = self.data[indices]

    def pearson_matrix(self, n_dim:int, normal=False):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        if n_dim == 0:
            self.data = self.data.transpose(0,2,1)
        graph = np.zeros((dim, dim))
        for i in range(dim):
            seq_i = self.data[:, :, i].flatten()
            for j in range(i, dim):
                seq_j = self.data[:, :, j].flatten()
                p = scipy.stats.pearsonr(seq_i, seq_j)[0]
                if normal:
                    try: 
                        p = p+1/2
                    except:
                        p = 0.5
                graph[i,j] = p
                graph[j,i] = p
        return graph

    def cosine_similarity_matrix(self, n_dim:int, normal=False):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        if n_dim == 0:
            self.data = self.data.transpose(0,2,1)
        graph = np.zeros((dim, dim))
        for i in range(dim):
            seq_i = self.data[:, :, i]
            for j in range(i, dim):
                seq_j = self.data[:, :, j]
                sim = cosine_similarity(seq_i, seq_j)[0][0]
                if normal:
                    try: 
                        sim = sim+1/2
                    except:
                        sim = 0.5
                graph[i,j] = sim
                graph[j,i] = sim
        return graph
    
    def load_pkl_graph(self, pkl_path:str):
        graph = pickle.load(open(pkl_path, 'rb'))
        return graph
    
    #method to generate adjacency matrix in datasets with spatial information(PEMS, METR-LA, etc.)
    def space_adj_matrix(self, dist_path, n_vertex):
        import csv
        with open(dist_path, 'r') as f:
            parse = csv.reader(f)
            next(parse, None)
            dist=list(parse)
    
            adj_matrix=np.zeros((n_vertex,n_vertex))
            for i in range(0,len(dist)):
                fr=int(dist[i][0])
                t=int(dist[i][1])
                d=float(dist[i][2])
                print(fr,t,d)
                adj_matrix[fr][t]=d
            return adj_matrix

    #generate weighted adjacency matrix
    def preprocess_adj_matrix(self,adj_matrix, sigma, threshold=0.1):
        tmp=np.multiply(adj_matrix, adj_matrix)
        wt_matrix=np.exp(tmp/(-sigma*sigma))
        wt_matrix[wt_matrix<threshold]=0
        return wt_matrix

