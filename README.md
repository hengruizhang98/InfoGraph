# InfoGraph
(Still under construction)

This DGL example implements the model proposed in the paper [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](https://arxiv.org/abs/1908.01000).

Paper link: https://arxiv.org/abs/1908.01000

Author's code: https://github.com/fanyun-sun/InfoGraph

Contributor: Hengrui Zhang ([@hengruizhang98](https://github.com/hengruizhang98))

## Dependecies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.6.0

## Dataset

Unsupervised Graph Classification Setting:

|   Dataset    | MUTAG | PTC  | **IMDBBINARY** | **IMDBMULTI** | **REDDITBINARY** | **REDDITMULTI5K** |
| :----------: | :---: | :--: | :------------: | :-----------: | :--------------: | :---------------: |
|   #Graphs    |       |      |                |               |                  |                   |
| # Avg. Nodes |       |      |                |               |                  |                   |
| # Avg. Edges |       |      |                |               |                  |                   |

Semi-supervised Graph Regression Setting:

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



## Examples

Train a model which follows the original hyperparameters on different datasets.

```bash
# Cora:
python main.py --dataname cora --gpu 0 --lam 1.0 --tem 0.5 --order 8 --sample 4 --input_droprate 0.5 --hidden_droprate 0.5 --dropnode_rate 0.5 --hid_dim 32 --early_stopping 100 --lr 1e-2  --epochs 2000

```

### Performance

The hyperparameter setting in our implementation is identical to that reported in the paper.

|      Dataset      |     MUTAG      |      PTC       |        ax        |
| :---------------: | :------------: | :------------: | :--------------: |
| Accuracy Reported | **85.4(±0.4)** | **75.4(±0.4)** |    82.7(±0.6)    |
|  This repository  |  85.33(±0.41)  |  75.36(±0.36)  | **82.90(±0.66)** |



