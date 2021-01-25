import torch as th

import dgl
from dgl.data import GINDataset
# form dgl.data import QM9Dataset

from torch.utils.data import DataLoader

from model import InfoGraph
from evaluate_embedding import evaluate_embedding

import argparse

def argument():
    parser = argparse.ArgumentParser(description='InfoGraph')

    # data source params
    parser.add_argument('--dataname', type=str, default='MUTAG', help='Name of dataset. Choose from {MUTAG}')

    # training params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, default:-1, using CPU.')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')

    # model params
    parser.add_argument('--n_layers', type=int, default=3, help='Number21 of graph convolution layers before each pooling')
    parser.add_argument('--hid_dim', type=int, default=32, help='Hidden layer dimensionalities')

    args = parser.parse_args()
    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    return args

    
def collate(samples):
    ''' an auxiliary function for building graph dataloader'''

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    n_nodes = batched_graph.num_nodes()

    batch = th.zeros(n_nodes).long()
    N = 0
    id = 0
    for graph in graphs:
        N_next = N + graph.num_nodes()
        batch[N:N_next] = id
        N = N_next
        id += 1
    batched_graph.ndata['graph_id'] = batch
    batched_graph.ndata['attr'] = batched_graph.ndata['attr'].to(th.float32)

    return batched_graph, batched_labels


if __name__ == '__main__':

    args = argument()
    print('device:', args.device)
    dataset = GINDataset(args.dataname, False)

    graphs, labels = map(list, zip(*dataset))
    wholegraph = dgl.batch(graphs)
    wholegraph.ndata['attr'] = wholegraph.ndata['attr'].to(th.float32)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            collate_fn=collate,
                            drop_last=False,
                            shuffle=True)

    in_dim = dataset[0][0].ndata['attr'].shape[1]

    model = InfoGraph(in_dim, args.hid_dim, args.n_layers)
    model = model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
 
    
    print('===== Before training =====')
    emb = model.get_embedding(wholegraph)
    res = evaluate_embedding(emb, labels)
    print('logreg {:4f}, svc {:4f}'.format(res[0], res[1]))
    
    best_logreg = 0
    best_svc = 0
    best_epoch = 0
    best_loss = 0
    
    for epoch in range(1, args.epochs):
        loss_all = 0
        model.train()
    
        for graph, label in dataloader:
            graph = graph.to(args.device)
            n_graph = label.shape[0]
    
            optimizer.zero_grad()
            loss = model(graph)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * n_graph
    
        print('Epoch {}, Loss {:.4f}'.format(epoch, loss_all / len(dataloader)))
    
        if epoch % 1 == 0:
            model.eval()
            emb = model.get_embedding(wholegraph)
            res = evaluate_embedding(emb, labels)
            
            if res[0] > best_logreg:
                best_logreg = res[0]
            if res[1] > best_svc:
                best_svc = res[1]
                best_epoch = epoch
                best_loss = loss_all
    
        if epoch % 5 == 0:
            print('logreg {:4f}, best svc {:4f}, best_epoch: {}, best_loss: {}'.format(res[0], best_svc, best_epoch,
                                                                                       best_loss))
    print('Training End')
    print('best svc {:4f}'.format(best_svc))
