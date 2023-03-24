import math
import torch
import torch.nn.functional as F
import torch.nn as nn

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
    A_tilde = A + torch.eye(A.shape[0])
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

    def __init__(self, input_dim, output_dim, A, step_size, use_W=True, nonlinear=False):
        super(GRAFFLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_W = use_W
        self.nonlinear = nonlinear
        self.A = A
        self.step_size = step_size
        # self.adj_norm = sym_norm_adj(A)
        self.adj_norm = A + torch.eye(A.shape[0])
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        self.W = nn.parameter.Parameter(torch.empty((input_dim, input_dim)))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x):
        # return x + self.step_size * self.adj_norm @ x @ (self.W + self.W.T)
        return x + self.step_size * self.adj_norm @ x @ self.W


class GNN(nn.Module):
    """Simple encoder decoder GNN model using the various conv layers implemented by students

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        hidden_dim (int): Dimensionality of the hidden feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        time (int):
        step_size (int):
        A (torch.Tensor): 2-D adjacency matrix
        conv_type (str):
    """

    def __init__(self, input_dim, hidden_dim, output_dim, T, step_size, A):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = int(T // step_size)
        self.A = A

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.53),
        )
        self.conv = nn.Sequential(
            GRAFFLayer(hidden_dim, hidden_dim, A, step_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.34),
        )

        self.evolution = []

    def forward(self, x):
        self.evolution = []
        x = self.encoder(x)

        for _ in range(self.layers):
            self.evolution.append(x)
            x = self.conv(x)

        self.evolution.append(x)
        x = self.decoder(x)

        y_hat = F.log_softmax(x, dim=1)
        return y_hat

    @staticmethod
    def for_dataset(dataset, hidden_dim, T, step_size):
        A = to_dense_adj(dataset.edge_index)[0]
        return GNN(
            input_dim=dataset.x.shape[-1],
            hidden_dim=hidden_dim,
            output_dim=dataset.y.shape[-1],
            T=T,
            step_size=step_size,
            A=A
        )
