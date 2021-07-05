import torch
import torch.nn.functional as F
import random
import argparse
import time

from util import Logger, str2bool
from elasticgnn import get_model
from train_eval import train, test

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='ElasticGNN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='ElasticGNN')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=0, help='default: fix split')
    parser.add_argument('--seed', type=int, default=12321312)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=3)
    parser.add_argument('--lambda2', type=float, default=3)
    parser.add_argument('--L21', type=str2bool, default=True)
    parser.add_argument('--ptb_rate', type=float, default=0)

    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    return args

def main():
    args = parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')

    logger = Logger(args.runs * random_split_num)

    total_start = time.perf_counter()

    if 'adv' in args.dataset:
        from dataset_adv import get_dataset
    else:
        from dataset import get_dataset

    ## data split
    for split in range(random_split_num):
        dataset, data, split_idx = get_dataset(args, split)
        train_idx = split_idx['train']
        data = data.to(device)
        print("Data:", data)
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()

        model = get_model(args, dataset)
        print(model)

        ## multiple run for each split
        for run in range(args.runs):
            runs_overall = split * args.runs + run
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            t_start = time.perf_counter()
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                loss = train(model, data, train_idx, optimizer)
                result = test(model, data, split_idx)
                logger.add_result(runs_overall, result)
                    
                if args.log_steps > 0:
                    if epoch % args.log_steps == 0:
                        train_acc, valid_acc, test_acc = result
                        print(f'Split: {split + 1:02d}, '
                              f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}% '
                              f'Test: {100 * test_acc:.2f}%')

            t_end = time.perf_counter()
            duration = t_end - t_start
            if args.log_steps > 0:
                print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'), 'time: ', duration)
                logger.print_statistics(runs_overall)

    total_end = time.perf_counter()
    total_duration = total_end - total_start
    print('total time: ', total_duration)
    logger.print_statistics()

if __name__ == "__main__":
    main()

