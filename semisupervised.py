import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dgl

from process_data import QM9Dataset
from model import InfoGraphS
import argparse


def argument():
    parser = argparse.ArgumentParser(description='InfoGraph')

    # data source params
    parser.add_argument('--target', type=int, default=0, help='Choose regression task}')
    parser.add_argument('--train_num', type=int, default=5000, help='Number of training set')

    # training params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, default:-1, using CPU.')
    parser.add_argument('--epochs', type=int, default=400, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    # model params
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden layer dimensionalities')
    parser.add_argument('--reg', type=int, default=0.001, help='regularization coefficent')

    args = parser.parse_args()
    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    return args


def collate(samples):
    ''' an auxiliary function for building graph dataloader'''

    graphs, targets = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_targets = th.stack(targets)
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

    return batched_graph, batched_targets


if __name__ == '__main__':

    args = argument()
    print(args)

    device = 'cuda:0'

    dataset = QM9Dataset()

    graphs = dataset.graphs
    N = len(graphs)
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)

    val_idx = all_idx[:10000]
    test_idx = all_idx[10000:20000]
    train_idx = all_idx[20000:20000 + args.train_num]

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    unsup_idx = all_idx[20000:]
    unsup_data = [dataset[i] for i in unsup_idx]

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              collate_fn=collate,
                              drop_last=False,
                              shuffle=True)

    unsup_loader = DataLoader(unsup_data,
                              batch_size=args.batch_size,
                              collate_fn=collate,
                              drop_last=False,
                              shuffle=True)

    in_dim = dataset[0][0].ndata['attr'].shape[1]

    tar = 1
    patience = 0
    print('======== target = {} ========'.format(tar))
    args.target = tar
    mean = dataset[:][1][:, args.target].mean().item()
    std = dataset[:][1][:, args.target].std().item()

    model = InfoGraphS(in_dim, args.hid_dim)
    model = model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=6, min_lr=0.000001
    )

    sup_loss_all = 0
    unsup_loss_all = 0
    consis_loss_all = 0

    best_val_error = 100
    for epoch in range(args.epochs):
        ''' Training '''
        model.train()
        lr = scheduler.optimizer.param_groups[0]['lr']

        iteration = 0
        sup_loss_all = 0
        unsup_loss_all = 0
        consis_loss_all = 0
        for data1, data2 in zip(train_loader, unsup_loader):
            graph1, target1 = data1
            graph2, target2 = data2

            graph1 = graph1.to(args.device)
            graph2 = graph2.to(args.device)

            target1 = (target1[:, args.target] - mean) / std
            target2 = (target2[:, args.target] - mean) / std

            target1 = target1.to(args.device)
            target2 = target2.to(args.device)

            optimizer.zero_grad()

            nfeat1 = graph1.ndata.pop('attr')
            efeat1 = graph1.edata.pop('edge_attr')
            graph_id1 = graph1.ndata.pop('graph_id')

            nfeat2 = graph2.ndata.pop('attr')
            efeat2 = graph2.edata.pop('edge_attr')
            graph_id2 = graph2.ndata.pop('graph_id')

            sup_loss = F.mse_loss(model.sup_pred(graph1, nfeat1, efeat1), target1)

            unsup_loss, consis_loss = model.generate_loss(graph2, nfeat2, efeat2, graph_id2)

            loss = sup_loss + unsup_loss + args.reg * consis_loss

            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            consis_loss_all += consis_loss.item()

            optimizer.step()

        print('Epoch: {}, Sup_Loss: {:4f}, Unsup_loss: {:.4f}, Consis_loss: {:.4f}' \
              .format(epoch, sup_loss_all, unsup_loss_all, consis_loss_all))

        model.eval()

        val_error = 0
        test_error = 0

        for i in range(100):
            val_graphs = [a[0] for a in val_data[i * 100:(i + 1) * 100]]
            val_targets = [a[1][args.target] for a in val_data[i * 100:(i + 1) * 100]]
            val_target = ((th.stack(val_targets) - mean) / std).to(args.device)
            val_graph = dgl.batch(val_graphs).to(args.device)
            val_nfeat = val_graph.ndata['attr']
            val_efeat = val_graph.edata['edge_attr']
            val_error += (model.sup_pred(val_graph, val_nfeat, val_efeat) * std - val_target * std).abs().sum().item()

        val_error = val_error / 10000
        scheduler.step(val_error)

        if val_error < best_val_error:
            best_val_error = val_error
            patience = 0

            for i in range(100):
                test_graphs = [a[0] for a in test_data[i * 100:(i + 1) * 100]]
                test_targets = [a[1][args.target] for a in test_data[i * 100:(i + 1) * 100]]
                test_target = ((th.stack(test_targets) - mean) / std).to(args.device)
                test_graph = dgl.batch(test_graphs).to(args.device)
                test_nfeat = test_graph.ndata['attr']
                test_efeat = test_graph.edata['edge_attr']
                test_error += (model.sup_pred(test_graph, test_nfeat,
                                              test_efeat) * std - test_target * std).abs().sum().item()
            test_error = test_error / 10000
            best_test_error = test_error
        else:
            patience += 1

        print('Epoch: {}, LR: {}, best_val_error: {:.4f}, val_error: {:.4f}, test_error: {:.4f}' \
              .format(epoch, lr, best_val_error, val_error, best_test_error))

        # if patience == 20:
        #     print('training ends')
        #     break

    with open('semisupervised.log', 'a+') as f:
        f.write('{},{},{}\n'.format(tar, val_error, test_error))
