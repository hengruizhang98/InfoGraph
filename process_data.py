import os
import torch as th

import dgl
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset

class QM9Dataset():

    def __init__(self, data_dir = 'qm9/'):
        super(QM9Dataset, self).__init__()
        
        self.save_path = data_dir
        self.load()
        
    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_path, 'qm9.bin'))
        self.graphs = graphs
        self.targets = label_dict['targets']
            
    def num_labels(self):
        return self.label.shape[1]

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


    def __len__(self):
        return len(self.graphs)


def transforms(graph):
    
    n_nodes = graph.num_nodes()
    row = th.arange(n_nodes, dtype = th.long)
    col = th.arange(n_nodes, dtype = th.long)

    row = row.view(-1,1).repeat(1, n_nodes).view(-1)
    col = col.repeat(n_nodes)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    idx = src * n_nodes + dst
    size = list(graph.edata['edge_attr'].size())
    size[0] = n_nodes * n_nodes
    edge_attr = graph.edata['edge_attr'].new_zeros(size)
    
    edge_attr[idx] = graph.edata['edge_attr']
    
    pos = graph.ndata['pos']
    dist = th.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
    
    new_edge_attr =  th.cat([edge_attr, dist.type_as(edge_attr)], dim = -1)
    
    new_graph = dgl.graph((src,dst))
    new_graph.ndata['attr'] = graph.ndata['attr']
    new_graph.edata['edge_attr'] = new_edge_attr
    
    return new_graph
