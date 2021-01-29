# InfoGraph
(Still under construction)

This DGL example implements the model proposed in the paper [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](https://arxiv.org/abs/1908.01000).

Paper link: https://arxiv.org/abs/1908.01000

Author's code: https://github.com/fanyun-sun/InfoGraph

Contributor: Hengrui Zhang ([@hengruizhang98](https://github.com/hengruizhang98))

## Dependecies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.5.3

## Dataset

Unsupervised Graph Classification Setting (dgl.data.GINDataset):

|   Dataset    | MUTAG | PTC  | **IMDBBINARY** | **IMDBMULTI** | **REDDITBINARY** | **REDDITMULTI5K** |
| :----------: | :---: | :--: | :------------: | :-----------: | :--------------: | :---------------: |
|   #Graphs    |       |      |                |               |                  |                   |
| # Avg. Nodes |       |      |                |               |                  |                   |
| # Avg. Edges |       |      |                |               |                  |                   |

Semi-supervised Graph Regression Setting (QM9Dataset):

| Dataset | # Graphs | # Avg. Nodes | #  Avg. Edges |
| ------- | -------- | ------------ | ------------- |
| QM9     |          |              |               |




## Arguments

#### 	Unsupervised

###### Dataset options

```
--dataname          str     The graph dataset name.             Default is 'MUTAG'.
```

###### GPU options

```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Training options

```
--epochs           int     Number of training epochs.             Default is 20.
--batch_size       int     Size of a training batch               Default is 128.
--lr               float   Adam optimizer learning rate.          Default is 0.01.
```

###### Model options

```
--n_layers         int     Number of GIN layers.                  Default is 3.
--hid_dim          int     Dimension of hidden layer.             Default is 32.
```

#### 

#### Semi-supervised


###### Dataset options

```
--dataname         str     The graph dataset name.             Default is 'MUTAG'.
--target           int     The id of regression task.          Default is 0.
--train_num        int     Number of training examples.        Default is 5000.
```

###### GPU options

```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Training options

```
--epochs           int     Number of training epochs.             Default is 200.
--batch_size       int     Size of a training batch               Default is 20.
--lr               float   Adam optimizer learning rate.          Default is 0.001.
```

###### Model options

```
--hid_dim          int     Dimension of hidden layer.             Default is 64.
--reg              int     Regularization weight                  Default is 0.001.
```

## Examples

Training and testing unsupervised model on MUTAG.(We recommend using cpu)
```bash
# MUTAG:
python unsupervised.py --dataname MUTAG --n_layers 4 --hid_dim 32
```
Training and testing semi-supervised model on QM9-$\mu$.
```bash
# QM9:
python semisupervised.py --gpu 0 --target 0
```

### Performance

The hyperparameter setting in our implementation is identical to that reported in the paper.

Unsupervised Setting

|      Dataset      | MUTAG | PTC  | REDDIT-B | REDDIT-M | IMDB-B | IMDB-M |
| :---------------: | :---: | :--: | :------: | -------- | ------ | ------ |
| Accuract Reported |       |      |          |          |        |        |
|  This repository  |       |      |          |          |        |        |



Semi-supervised setting

|      Task       | Mu, $\mu$ (0)  | Alpha, $\alpha$ (1) |      |      |      |
| :-------------: | :------------: | :-----------------: | :--: | ---- | ---- |
|  RMSE Reported  | **85.4(±0.4)** |   **75.4(±0.4)**    |      |      |      |
| Author's codes  |  85.33(±0.41)  |    75.36(±0.36)     |      |      |      |
| This repository |                |                     |      |      |      |

