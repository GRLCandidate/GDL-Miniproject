import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn

from torch_geometric.utils import to_dense_adj


def dirichlet_energy(X, adj_norm):
    # X is a matrix of shape (num_nodes, feature_channels)
    # adj_norm is a torch sparse coo tensor
    L_norm = torch.eye(X.shape[0]) - adj_norm
    energy = torch.trace(X.T @ L_norm @ X)
    return energy


def sym_norm_adj(A):
    """ Create the symmetric normalised adjacency from the dense adj matrix A"""
    # This should return a sparse adjacency matrix. (torch sparse coo tensor format)
    #A_tilde = A + torch.eye(A.shape[0])
    A_tilde = A
    D_tilde = torch.diag(A_tilde.sum(dim=0))
    D_tilde_inv_sqrt = D_tilde.pow(-1 / 2)
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
    A_tilde = A_tilde.to_sparse()
    D_tilde_inv_sqrt = D_tilde_inv_sqrt.to_sparse()
    adj_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
    return adj_norm


class GRAFFLayer(nn.Module):
    """GRAFF layer.

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """

    def __init__(self, input_dim, output_dim, step_size):
        super(GRAFFLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_size = step_size

        self.Omega = nn.parameter.Parameter(torch.empty((input_dim,)))
        self.W = nn.parameter.Parameter(torch.empty((input_dim, input_dim)))
        self.W_tilde = nn.parameter.Parameter(torch.empty((input_dim, input_dim)))

        nn.init.ones_(self.Omega)
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.W_tilde)

    def forward(self, adj_norm, x, x0):
        # return x + self.step_size * self.adj_norm @ x @ (self.W + self.W.T)
        residual = x * self.Omega
        convo = adj_norm @ x @ (self.W + self.W.T)
        initial = x0 @ self.W_tilde

        # print(f'x: {x.shape} x0: {x0.shape} residual: {residual.shape} convo: {convo.shape} initial: {initial.shape}')
        return x + self.step_size * (-residual + convo - initial)


class GRAFFNetwork(nn.Module):
    """Simple encoder decoder GNN model using the various conv layers implemented by students

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        hidden_dim (int): Dimensionality of the hidden feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        T (int):
        step_size (float):
    """

    def __init__(self, input_dim, hidden_dim, output_dim, T, step_size, dropout):
        super(GRAFFNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = int(T // step_size)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.conv = GRAFFLayer(hidden_dim, hidden_dim, step_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

    def forward(self, data, debug=False):
        x0 = self.encoder(data.x)
        x = x0

        for _ in range(self.layers):
            x = self.conv(data.adj_norm, x, x0)

        x = self.decoder(x)

        y_hat = F.softmax(x, dim=1)
        return y_hat

    @staticmethod
    def for_dataset(dataset, hidden_dim, T, step_size, dropout):
        return GRAFFNetwork(
            input_dim=dataset.x.shape[-1],
            hidden_dim=hidden_dim,
            output_dim=2,
            T=T,
            step_size=step_size,
            dropout=dropout,
        )


class MultiGRAFFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, T, step_size, dropout, ev_trans_shared):
        super(MultiGRAFFNetwork, self).__init__()
        self.num_convs_per_graff = int(T // step_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm((hidden_dim,)),
            nn.Dropout(dropout),
        )

        self.ev_trans_shared = ev_trans_shared
        if ev_trans_shared:
            self.ev_trans = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm((hidden_dim,)),
                nn.LeakyReLU(),
            )
        else:
            self.ev_trans = []
            for _ in range(num_layers):
                self.ev_trans.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm((hidden_dim,)),
                    nn.LeakyReLU(),
                ))
        convs = []
        for _ in range(num_layers):
            convs.append(GRAFFLayer(hidden_dim, hidden_dim, step_size))

        self.convs = nn.Sequential(*convs)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, data, debug=False):
        x0 = self.encoder(data.x)
        x = x0

        if self.ev_trans_shared:
            for conv in self.convs:
                for _ in range(self.num_convs_per_graff):
                    x = conv(data.adj_norm, x, x0)
                x = self.ev_trans(x)
        else:
            for conv, trans in zip(self.convs, self.ev_trans):
                for _ in range(self.num_convs_per_graff):
                    x = conv(data.adj_norm, x, x0)
                x = trans(x)

        x = self.decoder(x)
        y = F.softmax(x, dim=1)
        if debug and False:
            print(f'x: {x[:10,:]}')
            print(f'y: {y[:10,:]}')
        return y

    @staticmethod
    def for_dataset(dataset, **kwargs):
        return MultiGRAFFNetwork(
            input_dim=dataset.x.shape[-1],
            output_dim=2,
            **kwargs
        )


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        eps = 1e-10
        intersection = torch.sum(torch.mul(pred[:, 1], target))
        union = torch.sum(pred[:, 1] + target)
        return 1 - 2 * (intersection + eps) / (union + eps)


class ClassicGNN(nn.Module):
    def __init__(self, base):
        super(ClassicGNN, self).__init__()
        self.base = base

    def forward(self, data):
        y = self.base(data.x, data.edge_index)
        y_hat = F.softmax(y, dim=1)
        return y_hat

    @staticmethod
    def for_dataset(dataset, hidden_dim, num_layers):
        return ClassicGNN(gnn.GraphSAGE(
            in_channels=dataset.x.shape[-1],
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=2,
        ))
