import networkx as nx
import torch_geometric
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric import utils as U
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from ads.models import sym_norm_adj


def get_data(num_datapoints, n, num_colors, expression_length, seed=None, **kwargs):
    if seed is not None:
        datapoints = [mk_datapoint(n, num_colors, expression_length, seed=seed + i, **kwargs) for i in range(num_datapoints)]
    else:
        datapoints = [mk_datapoint(n, num_colors, expression_length, **kwargs) for _ in range(num_datapoints)]

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    train, val, test = random_split(datapoints, [0.6, 0.3, 0.1], gen)
    dls = DataLoader(train), DataLoader(val), DataLoader(test)
    for dl in dls:
        dl.description = {
            'num_datapoints': num_datapoints,
            'n': n,
            'num_colors': num_colors,
            'expression_length': expression_length,
            'seed': seed,
            'kwargs': kwargs,
        }
    return dls


def add_colors(data, colors, expression_length, seed):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    color_idx = torch.randint(0, colors, (data.num_nodes,), generator=generator)
    color = torch.eye(colors)[color_idx]
    data.x = color

    y = torch.ones(data.num_nodes).int()
    adj_matrix = U.to_dense_adj(data.edge_index).int()
    for c in range(expression_length):
        adj_nodes = (adj_matrix @ y).clip(max=1).int()
        y = adj_nodes & (color_idx == c)
        y = y.squeeze()

    data.y = y.to(torch.long)


def mk_datapoint(n, colors, expression_length, out_degree=2, seed=None):
    assert expression_length <= colors
    graph = nx.random_k_out_graph(n, out_degree, 3, self_loops=False, seed=seed).to_undirected()
    data = torch_geometric.utils.from_networkx(graph)

    add_colors(data, colors, expression_length, seed)
    data.adj_norm = sym_norm_adj(U.to_dense_adj(data.edge_index)[0])

    return data


def to_networkx(data):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(data.num_nodes))
    for i, j in data.edge_index.t():
        graph.add_edge(i.item(), j.item())
    return graph


def draw_graph(data, pos=None, seed=None, axis=None):
    graph = to_networkx(data)
    if pos is None:
        pos = nx.spring_layout(graph, seed=seed)

    node_colors = torch.argmax(data.x, dim=1)
    colors = np.array(['red', 'green', 'blue', 'purple', 'grey', 'black', 'white', 'orange'])
    color_str = colors[node_colors.detach().numpy()]

    size = data.y * 200 + 100

    nx.draw(
        graph,
        edge_color='grey',
        node_color=color_str,
        node_size=size,
        arrows=False,
        pos=pos,
        with_labels=False,
        node_shape='o',
        ax=axis,
    )
    return pos


if __name__ == "__main__":
    data, _, _ = get_data(100, 40, 5, 5, out_degree=3, seed=1045966)
    for d in data:
        draw_graph(d, seed=1045966)
        plt.show()
