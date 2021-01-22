import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dgl

from process_data import QM9Dataset
from model import InfoGraphS
from evaluate_embedding import evaluate_embedding

import argparse


def argument():
    parser = argparse.ArgumentParser(description='InfoGraph')

    # data source params
    parser.add_argument('--target', type=int, default=0, help='Choose regression task}')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test set')
    parser.add_argument('--train_num', type=int, default=50000, help='Number of training set')
    
    # training params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, default:-1, using CPU.')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')

    # model params
    parser.add_argument('--hid_dim', type=int, default=32, help='Hidden layer dimensionalities')
    parser.add_argument('--reg', type=int, default=0.1, help='regularization coefficent')


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

    device = 'cuda:0'

    dataset = QM9Dataset()

    mean = dataset[:][1][:, args.target].mean().item()
    std = dataset[:][1][:,args.target].std().item()
    
    graphs = dataset.graphs
    N = len(graphs)
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    
    val = all_idx[:10000]
    test_idx = all_idx[10000:20000]
    train_idx = all_idx[20000:20000+args.train_num]


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

    val_loader = DataLoader(val_data,
                        batch_size=args.batch_size,
                        collate_fn=collate,
                        drop_last=False,
                        shuffle=True)

    test_loader = DataLoader(test_data,
                        batch_size=args.batch_size,
                        collate_fn=collate,
                        drop_last=False,
                        shuffle=True)

    unsup_loader = DataLoader(unsup_data,
                        batch_size=args.batch_size,
                        collate_fn=collate,
                        drop_last=False,
                        shuffle=True)

    # train_idx = all_idx[: int(N*args.train_ratio)]
    # val_idx = all_idx[int(N*args.train_ratio) : int(N*(args.train_ratio + args.val_ratio))]
    # test_idx = all_idx[int(N*(args.train_ratio + args.val_ratio)): int(N*(args.train_ratio + args.val_ratio + args.test_ratio))]

    # train_data = [dataset[i] for i in train_idx]
    # val_data = [dataset[i] for i in val_idx]
    # test_data = [dataset[i] for i in test_idx]


    in_dim = dataset[0][0].ndata['attr'].shape[1]

    model = InfoGraphS(in_dim, args.hid_dim)
    model = model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    
    sup_loss_all = 0
    unsup_loss_all = 0
    consis_loss_all = 0

    for epoch in range(args.epochs):
        
        ''' Training '''
        iter = 0
        model.train()
        for labeled_data, unlabeled_data in zip(train_loader, unsup_loader):

            labeled_graph, labeled_targets = labeled_data
            unlabeled_graph, unlabeled_targets = unlabeled_data

            labeled_target = ((labeled_targets[:,args.target] - mean) / std).to(args.device)
            # unlabeled_target = ((unlabeled_targets[:,args.target] - mean) / std).to(args.device)

            optimizer.zero_grad()
            sup_loss = F.mse_loss(model(labeled_graph), labeled_target)
            
            unsup_loss, consis_loss = model.unsup_forward(labeled_target)
            
            loss = sup_loss + unsup_loss + args.reg * consis_loss
            
            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            consis_loss_all += consis_loss.item()

            optimizer.step()

            print('Iter: {}, Sup_Loss: {:.4f}, Unsup_loss: {:.4f}, Consis_loss: {:.4f}' \
                  .format(iter, sup_loss, unsup_loss, consis_loss))

            iter += 1

        print('Epoch: {}, Sup_Loss: {:.4f}, Unsup_loss: {:.4f}, Consis_loss: {:.4f}'\
            .format(epoch, sup_loss_all, unsup_loss_all, consis_loss_all))



    # print('===== Before training =====')
    # emb = model.get_embedding(wholegraph)
    # res = evaluate_embedding(emb, labels)
    # print('logreg {:4f}, svc {:4f}'.format(res[0], res[1]))
    
    # best_logreg = 0
    # best_svc = 0
    # best_epoch = 0
    # best_loss = 0
    
    # for epoch in range(1, args.epochs):
    #     loss_all = 0
    #     model.train()
    
    #     for graph, label in dataloader:
    #         graph = graph.to(args.device)
    #         n_graph = label.shape[0]
    
    #         optimizer.zero_grad()
    #         loss = model(graph)
    #         loss.backward()
    #         optimizer.step()
    #         loss_all += loss.item() * n_graph
    
    #     print('Epoch {}, Loss {:.4f}'.format(epoch, loss_all / len(dataloader)))
    
    #     if epoch % 10 == 0:
            
    #         model.eval()
    #         emb = model.get_embedding(wholegraph)
    #         res = evaluate_embedding(emb, labels)
            
    #         if res[0] > best_logreg:
    #             best_logreg = res[0]
    #         if res[1] > best_svc:
    #             best_svc = res[1]
    #             best_epoch = epoch
    #             best_loss = loss_all
    
    #     if epoch % 10 == 0:
    #         print('logreg {:4f}, best svc {:4f}, best_epoch: {}, best_loss: {}'.format(res[0], best_svc, best_epoch,
    #                                                                                    best_loss))
    #         # print('logreg {:4f}, svc {:4f}'.format(res[0], res[1]))
    # print('Training End')
    # print('best svc {:4f}'.format(best_svc))
