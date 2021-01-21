import torch as th

import dgl
from dgl.data import GINDataset
# form dgl.data import QM9Dataset

from torch.utils.data import DataLoader

from model import InfoGraph
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--dataset', dest='dataset', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const',
                        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
                        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hid_dim', dest='hidden_dim', type=int, default=32,
                        help='')

    return parser.parse_args()


def collate(samples):
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
    batched_graph.ndata['batch'] = batch
    batched_graph.ndata['attr'] = batched_graph.ndata['attr'].to(th.float32)

    return batched_graph, batched_labels


if __name__ == '__main__':

    args = arg_parse()
    accuracies = {'logreg': [], 'svc': [], 'linearsvc': [], 'randomforest': []}
    epochs = 20
    log_interval = 1
    batch_size = 128
    device = 'cpu'
    n_epochs = 1000

    dataset = GINDataset('REDDITBINARY', False)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

    in_dim = dataset[0][0].ndata['attr'].shape[1]
    hid_dim = 32
    n_layer = 4

    # print(dataset[0][0])
    # print(dataset[1][0].ndata['attr'])

    model = InfoGraph(in_dim, hid_dim, n_layer)
    model = model.to(device)
    model.reset_parameters()

    print(model)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    print('===== Before training =====')
    emb, y = model.encoder.get_embedding(dataloader)
    res = evaluate_embedding(emb, y)
    print('logreg {:4f}, svc {:4f}'.format(res[0], res[1]))

    best_logreg = 0
    best_svc = 0
    best_epoch = 0
    best_loss = 0

    for epoch in range(1, 101):
        loss_all = 0
        model.train()

        for graph, label in dataloader:
            graph = graph.to(device)
            feat = graph.ndata.pop('attr')
            batch = graph.ndata.pop('batch')

            n_graph = label.shape[0]
            optimizer.zero_grad()
            loss = model(graph, feat, batch)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * n_graph

        print('Epoch {}, Loss {:.4f}'.format(epoch, loss_all / len(dataloader)))

        if epoch % 10 == 0:
            model.eval()
            emb, y = model.encoder.get_embedding(dataloader)
            res = evaluate_embedding(emb, y)
            if res[0] > best_logreg:
                best_logreg = res[0]
            if res[1] > best_svc:
                best_svc = res[1]
                best_epoch = epoch
                best_loss = loss_all

        if epoch % 10 == 0:
            print('logreg {:4f}, best svc {:4f}, best_epoch: {}, best_loss: {}'.format(res[0], best_svc, best_epoch,
                                                                                       best_loss))
            # print('logreg {:4f}, svc {:4f}'.format(res[0], res[1]))
    print('Training End')
    print('best svc {:4f}'.format(best_svc))
