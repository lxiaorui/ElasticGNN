import torch
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset

from util import index_to_mask, mask_to_index

def get_dataset(args, split, sparse=True, **kwargs):

    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None

    ## fix random seed for data split
    seeds_init = [12232231, 12232432, 2234234, 4665565, 45543345, 454543543, 45345234, 54552234, 234235425, 909099343]
    seeds = []
    for i in range(1, 20):
        seeds = seeds + [a*i for a in seeds_init]
    
    seed = seeds[split]

    if args.ogb: 
        dataset = get_ogbn_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = index_to_mask(split_idx['train'], data.x.shape[0])
        data.test_mask = index_to_mask(split_idx['test'], data.x.shape[0])
        data.val_mask = index_to_mask(split_idx['valid'], data.x.shape[0])
        return dataset, data, split_idx

    if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
        dataset = get_planetoid_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        if args.random_splits > 0:
            data = random_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed)
            print(f'random split {args.dataset} split {split}')

    elif args.dataset == "cs" or args.dataset == "physics":
        dataset = get_coauthor_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(f'random split {args.dataset} split {split}')

    elif args.dataset == "computers" or args.dataset == "photo":
        dataset = get_amazon_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(f'random split {args.dataset} split {split}')

    split_idx = {}
    split_idx['train'] = mask_to_index(data.train_mask)
    split_idx['valid'] = mask_to_index(data.val_mask)
    split_idx['test']  = mask_to_index(data.test_mask)

    return dataset, data, split_idx


def get_transform(normalize_features, transform):
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform
    return transform

def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_coauthor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Coauthor(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Amazon(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_ogbn_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = PygNodePropPredDataset(name, path)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def random_planetoid_splits(data, num_classes, seed):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
    return data

def random_coauthor_amazon_splits(data, num_classes, seed):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data