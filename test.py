import esan.data

import torch_geometric

from esan.data import TUDataset
from esan.models import DSSnetwork
from esan.conv import GraphConv

SEED = 1045966
REDDIT_B = TUDataset("./datasets/reddit-binary", 'REDDIT-BINARY')
MUTAG = TUDataset("./datasets/mutag/ego_nets", 'MUTAG', pre_transform=esan.data.NodeDeleted())
split = MUTAG.separate_data(SEED, 0)
m_train_dl = torch_geometric.loader.DataLoader(MUTAG[split['train']], batch_size=16, follow_batch=['subgraph_idx'])
model = DSSnetwork(
    num_layers=1,
    in_dim=7,
    emb_dim=16,
    num_tasks=2,
    feature_encoder=lambda x: x,
    GNNConv=GraphConv
)
for d in m_train_dl:
    if d.x.shape[0] == 1 or d.batch[-1] == 0:
        pass

    d.edge_attr = None
    #num_nodes_per_subgraph = torch_scatter.scatter_add(torch.ones(d.num_nodes), d.batch)
    #d.num_nodes_per_subgraph = num_nodes_per_subgraph
    model(d)
    break
