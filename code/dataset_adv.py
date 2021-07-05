import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T

from deeprobust.graph.data import Dataset as DeepRobust_Dataset
from deeprobust.graph.data import PrePtbDataset as DeepRobust_PrePtbDataset
from torch_geometric.data import Data

from util import index_to_mask, mask_to_index
from dataset import get_transform


def get_dataset(args, split, sparse=True):

    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None

    assert args.dataset in ["Cora-adv", "CiteSeer-adv", "PubMed-adv", "Polblogs-adv"], 'dataset not supported'
    if args.dataset == "Cora-adv":
        dataset = get_adv_dataset('cora', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)
    
    elif args.dataset == "CiteSeer-adv":
        dataset = get_adv_dataset('citeseer', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)
    
    elif args.dataset == "PubMed-adv":
        dataset = get_adv_dataset('pubmed', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)

    elif args.dataset == "Polblogs-adv":
        dataset = get_adv_dataset('polblogs', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)
    data = dataset.data

    split_idx = {}
    split_idx['train'] = mask_to_index(data.train_mask)
    split_idx['valid'] = mask_to_index(data.val_mask)
    split_idx['test']  = mask_to_index(data.test_mask)
    
    return dataset, data, split_idx


def get_adv_dataset(name, normalize_features=False, transform=None, ptb_rate=0.05, args=None):
    transform = get_transform(normalize_features, transform)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/attacked_graph')
    dataset = DeepRobust_Dataset(root=path, name=name, setting='nettack', require_mask=True, seed=15)
    dataset.x = torch.FloatTensor(dataset.features.todense())
    dataset.y = torch.LongTensor(dataset.labels)
    dataset.num_classes = dataset.y.max().item() + 1

    if ptb_rate > 0:
        perturbed_data = DeepRobust_PrePtbDataset(root='/tmp/',
                    name=name,
                    attack_method='meta',
                    ptb_rate=ptb_rate)
        edge_index = torch.LongTensor(perturbed_data.adj.nonzero())
    else:
        edge_index = torch.LongTensor(dataset.adj.nonzero())
    data = Data(x=dataset.x, edge_index=edge_index, y=dataset.y)
    
    clean_edge_index = torch.LongTensor(dataset.adj.nonzero())
    clean_data = Data(x=dataset.x, edge_index=clean_edge_index, y=dataset.y)

    if name == 'pubmed':
        ## for pubmed, we need to change the split in order to be consistent with the results in Pro-GNN paper
        # just for matching the results in the paper; seed details in https://github.com/ChandlerBang/Pro-GNN/issues/2
        from deeprobust.graph.utils import encode_onehot, get_train_val_test
        num_nodes = dataset.x.shape[0]
        idx_train, idx_val, idx_test = get_train_val_test(num_nodes,
                val_size=0.1, test_size=0.8, stratify=encode_onehot(dataset.labels), seed=15)
        data.train_mask = index_to_mask(idx_train, num_nodes)
        data.val_mask   = index_to_mask(idx_val, num_nodes)
        data.test_mask  = index_to_mask(idx_test, num_nodes)
    else:
        data.train_mask = torch.tensor(dataset.train_mask)
        data.val_mask   = torch.tensor(dataset.val_mask)
        data.test_mask  = torch.tensor(dataset.test_mask)

    dataset.data = transform(data)
    dataset.clean_data = transform(clean_data)
    dataset.data.clean_adj = dataset.clean_data.adj_t
    return dataset