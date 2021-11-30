import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv,global_mean_pool as gep,global_max_pool as gmp,global_add_pool as gap

class Model_Net(torch.nn.Module):
    def __init__(self, n_output=None,num_features=6165, output_dim=486, dropout=0.2,aa_num=21,hidden_dim = 128,net = None,pool=None):
        super(Model_Net, self).__init__()
        self.pool = pool
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1024)
        self.n_output = n_output
        #mol network
        self.feature_linear1 = torch.nn.Linear(num_features-aa_num,hidden_dim)
        self.feature_linear2 = torch.nn.Linear(aa_num, aa_num)
        if self.net == 'GCN':
            self.prot_conv1 = GCNConv(hidden_dim+aa_num, hidden_dim+aa_num)
            self.prot_conv2 = GCNConv(hidden_dim+aa_num, (hidden_dim+aa_num) * 2)
            self.prot_conv3 = GCNConv((hidden_dim+aa_num) * 2, (hidden_dim+aa_num) * 4)
        elif self.net == 'GAT':
            self.prot_conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
            self.prot_conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
            self.prot_conv3 = GATConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
        if '-' in self.pool:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim + aa_num) * 4 * 2, 1024)
        else:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim+aa_num) * 4, 1024)
        self.prot_fc_g2 = torch.nn.Linear(1024,  output_dim)



    def forward(self, data_prot):
        prot_x, prot_edge_index, prot_batch = data_prot.x, data_prot.edge_index, data_prot.batch
        prot_feature1 = self.relu(self.feature_linear1(prot_x[:,21:]))
        prot_feature2 = self.relu(self.feature_linear2(prot_x[:,:21]))
        prot_feature = torch.cat((prot_feature2,prot_feature1),1)
        x = self.prot_conv1(prot_feature,prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv2(x,prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv3(x,prot_edge_index)
        x = self.relu(x)
        if self.pool == 'gep':
            x = gep(x, prot_batch)
        elif self.pool == 'gmp':
            x = gmp(x, prot_batch)
        elif self.pool == 'gap':
            x = gap(x, prot_batch)
        elif self.pool == 'gap-gmp':
            x1 = gap(x, prot_batch)
            x2 = gmp(x, prot_batch)
            x = torch.cat((x1,x2),1)
        elif self.pool == 'gap-gep':
            x1 = gap(x, prot_batch)
            x2 = gep(x, prot_batch)
            x = torch.cat((x1, x2), 1)
        x = self.prot_fc_g1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.prot_fc_g2(x)
        x = self.sigmoid(x)

        return x




        #return out
