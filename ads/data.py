import networkx as nx
import torch_geometric
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric import utils as U
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split


def get_data(num_datapoints, n, num_colors, expression_length, seed=None, **kwargs):
    if seed is not None:
        datapoints = [mk_datapoint(n, num_colors, expression_length, seed=seed + i, **kwargs) for i in range(num_datapoints)]
    else:
        datapoints = [mk_datapoint(n, num_colors, expression_length, **kwargs) for _ in range(num_datapoints)]

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    train, val, test = random_split(datapoints, [0.6, 0.3, 0.1], gen)
    return DataLoader(train), DataLoader(val), DataLoader(test)


def mk_datapoint(n, colors, expression_length, out_degree=2, seed=None):
    assert expression_length <= colors
    graph = nx.random_k_out_graph(n, out_degree, 3, self_loops=False, seed=seed).to_undirected()
    data = torch_geometric.utils.from_networkx(graph)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    color_idx = torch.randint(0, colors, (n,), generator=generator)
    color = torch.eye(colors)[color_idx]
    data.x = color

    y = torch.ones(n).int()
    adj_matrix = U.to_dense_adj(data.edge_index).int()
    for c in range(expression_length):
        adj_nodes = (adj_matrix @ y).clip(max=1).int()
        y = adj_nodes & (color_idx == c)
        y = y.squeeze()

    data.y = y.to(torch.long)
    return data


def draw_graph(data, pos=None, seed=None):
    graph = U.to_networkx(data)
    if pos is None:
        pos = nx.spring_layout(graph, seed=seed)

    node_colors = torch.argmax(data.x, dim=1)
    colors = np.array(['red', 'green', 'blue', 'yellow', 'purple', 'grey', 'black', 'white', 'orange'])
    color_str = colors[node_colors.detach().numpy()]

    size = data.y * 300 + 200

    nx.draw(
        graph,
        edge_color='grey',
        node_color=color_str,
        node_size=size,
        pos=pos,
        with_labels=True,
    )
    return pos


if __name__ == "__main__":
    data, _, _ = get_data(100, 50, 6, 6, out_degree=3, seed=1045966)
    for d in data:
        draw_graph(d, seed=1045966)
        plt.show()
