import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ModuleList, Linear, GRU, ReLU, BatchNorm1d

from dgl.nn import GINConv, NNConv, Set2Set
from dgl.nn.pytorch.glob import SumPooling

from utils import global_global_loss_, local_global_loss_


''' Feedforward neural network'''

class FFNN(nn.Module):
    ''' 3-layer feed-forward neural networks with jumping connections'''

    def __init__(self, in_dim):
        super(FFNN, self).__init__()

        self.block = Sequential(Linear(in_dim, in_dim),
                                ReLU(),
                                Linear(in_dim, in_dim),
                                ReLU(),
                                Linear(in_dim, in_dim),
                                ReLU()
                                )

        self.jump_con = Linear(in_dim, in_dim)

    def forward(self, feat):
        block_out = self.block(feat)
        jump_out = self.jump_con(feat)

        out = block_out + jump_out

        return out


''' Unsupevised Setting '''

class GINEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layer):
        super(GINEncoder, self).__init__()

        self.n_layer = n_layer

        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(n_layer):
            if i == 0:
                n_in = in_dim
            else:
                n_in = hid_dim
            n_out = hid_dim
            block = Sequential(Linear(n_in, n_out),
                               ReLU(),
                               Linear(hid_dim, hid_dim)
                               )

            conv = GINConv(block, 'sum')
            bn = BatchNorm1d(hid_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        self.pool = SumPooling()


    def forward(self, graph, feat):

        xs = []
        x = feat
        for i in range(self.n_layer):
            x = F.relu(self.convs[i](graph, x))
            x = self.bns[i](x)

            xs.append(x)

        local_emb = th.cat(xs, 1)
        global_emb = self.pool(graph, local_emb)

        return global_emb, local_emb


class InfoGraph(nn.Module):
    def __init__(self, n_feature, hid_dim, n_layer):
        super(InfoGraph, self).__init__()

        self.n_layer = n_layer

        embedding_dim = hid_dim * n_layer

        self.encoder = GINEncoder(n_feature, hid_dim, n_layer)

        self.local_d = FFNN(embedding_dim)
        self.global_d = FFNN(embedding_dim)

    def get_embedding(self, graph):
        with th.no_grad():
            feat = graph.ndata['attr']
            global_emb, _ = self.encoder(graph, feat)

        return global_emb

    def forward(self, graph):
        feat = graph.ndata.pop('attr')
        graph_id = graph.ndata.pop('graph_id')

        global_emb, local_emb = self.encoder(graph, feat)

        global_h = self.global_d(global_emb)
        local_h = self.local_d(local_emb)

        measure = 'JSD'

        loss = local_global_loss_(local_h, global_h, graph_id, measure)

        return loss


''' Semisupevised Setting '''

class NNConvEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(NNConvEncoder, self).__init__()

        self.lin0 = Linear(in_dim, hid_dim)

        block = Sequential(Linear(5, 128), ReLU(), Linear(128, hid_dim * hid_dim))

        self.conv = NNConv(hid_dim, hid_dim, block, aggr = 'mean', root_weight = False)
        self.gru = GRU(hid_dim, hid_dim)

        self.set2set = Set2Set(hid_dim, n_iters=3, n_layers=1)

    def forward(self, graph, nfeat, efeat):

        out = F.relu(self.lin0(nfeat))
        h = out.unsqueeze(0)

        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(graph, out, efeat))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map.append(out)

        out = self.set2set(out)

        return out, feat_map[-1]


class InfoGraphS(nn.Module):
    def __init__(self, n_feature, hid_dim, n_layer):
        super(InfoGraphS, self).__init__()

        self.n_layer = n_layer


        self.sup_encoder = NNConvEncoder(n_feature, hid_dim)
        self.unsup_encoder = NNConvEncoder(n_feature, hid_dim)

        self.fc1 = Linear(2 * hid_dim, hid_dim)
        self.fc2 = Linear(hid_dim, 1)

        self.unsup_local_d = FFNN(hid_dim, hid_dim)
        self.unsup_global_d = FFNN(2 * hid_dim, hid_dim)

        self.sup_d = FFNN(2 * hid_dim, hid_dim)
        self.unsup_d = FFNN(2 * hid_dim, hid_dim)


    def forward(self, graph):
        nfeat = graph.ndata.pop('attr')
        efeat = graph.edata.pop('edge_attr')

        graph_id = graph.ndata.pop('graph_id')

        sup_global_emb, sup_local_emb = self.sup_encoder(graph, nfeat, efeat)

        sup_global_pred = self.fc2(F.relu(self.fc1(sup_global_emb)))
        sup_global_pred = sup_global_pred.view(-1)

        unsup_global_emb, unsup_local_emb = self.unsup_encoder(graph, nfeat, efeat)

        g_enc = self.unsup_global_d(unsup_global_emb)
        l_enc = self.unsup_local_d(unsup_local_emb)

        sup_g_enc = self.sup_d(sup_global_emb)
        unsup_g_enc = self.unsup_d(unsup_global_emb)

        # Calculate loss
        measure = 'JSD'
        unsup_loss = local_global_loss_(l_enc, g_enc, graph_id, measure)
        con_loss = global_global_loss_(sup_g_enc, unsup_g_enc, measure)

        return sup_global_pred, unsup_loss, con_loss
