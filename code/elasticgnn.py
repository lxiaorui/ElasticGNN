import torch
import torch.nn.functional as F
from torch.nn import Linear

from emp import EMP

class ElasticGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, **kwargs):
        super(ElasticGNN, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.prop = prop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj_t, data=data)
        return F.log_softmax(x, dim=1)

''' 
Model adapted from https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/mlp.py

'''


class ElasticGNN_OGB(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, prop):
        super(ElasticGNN_OGB, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.prop = prop

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data, **kwargs):
        x, adj_t, = data.x, data.adj_t
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = self.prop(x, adj_t, data=data)
        return torch.log_softmax(x, dim=-1)



def get_model(args, dataset):

    if 'adv' in args.dataset:
        data = dataset.data
    else:
        data = dataset[0]

    if args.ogb: 
        Model = ElasticGNN_OGB 
    else:
        Model = ElasticGNN

    prop =  EMP(K=args.K, 
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                L21=args.L21,
                cached=True)

    model = Model(in_channels=data.num_features, 
                       hidden_channels=args.hidden_channels, 
                       out_channels=dataset.num_classes, 
                       dropout=args.dropout, 
                       num_layers=args.num_layers, 
                       prop=prop).cuda()

    return model