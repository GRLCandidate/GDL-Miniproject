from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj


def get_data():
    ds = Planetoid('datasets/cora', 'Cora')[0]
    return ds
