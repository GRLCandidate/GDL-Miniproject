import itertools
import torch
import time

from ads import data
from ads.grid_search import grid_search
from ads.models import GNN, ClassicGNN


def train_good_graff(iterations, train_dl, val_dl):
    param_grid = {
        # "T": [2, 3, 4],
        # "step_size": [1, 0.5, 0.25, 0.125],
        "hidden_dim": [8],
        "T": [3],
        "step_size": [0.25],
    }
    return grid_search(1, train_dl, val_dl, GNN, param_grid, num_epochs=100, lr=0.0026)


def train_good_gcn(iterations, train_dl, val_dl):
    param_grid = {
        # "T": [2, 3, 4],
        # "step_size": [1, 0.5, 0.25, 0.125],
        "hidden_dim": [8],
        "T": [3],
        "step_size": [0.25],
    }
    return grid_search(1, train_dl, val_dl, ClassicGNN, param_grid, num_epochs=100, lr=0.0026)


def main():
    seed = itertools.count(1045966, 1)
    iterations = 1
    graph_size = 100
    num_datapoints = 100
    datasets = [
        data.get_data(num_datapoints, graph_size, 8, 1, seed=seed.__next__()),
        data.get_data(num_datapoints, graph_size, 8, 2, seed=seed.__next__()),
        # data.get_data(num_datapoints, graph_size, 8, 3, seed=seed.__next__()),
        # data.get_data(num_datapoints, graph_size, 8, 4, seed=seed.__next__()),
        # data.get_data(num_datapoints, graph_size, 8, 5, seed=seed.__next__()),
    ]

    for idx, (train_dl, val_dl, test_dl) in enumerate(datasets):
        if False:
            r = train_good_graff(iterations, train_dl, val_dl)
            print(f'best parameter choice for graff was {r["args"]}, achieving avg val acc={r["acc"]:.4f}±{r["std"]:.4f} (models: {r["models"]})')
            torch.save(r, f'results/{int(time.time())}-{idx}-results-graff.pt')
            print('\n')

        r = train_good_gcn(iterations, train_dl, val_dl)
        print(f'best parameter choice for gcn was {r["args"]}, achieving avg val acc={r["acc"]:.4f}±{r["std"]:.4f} (models: {r["models"]})')
        torch.save(r, f'results/{int(time.time())}-{idx}-results-gcn.pt')
        print('\n')

    # Generate a bunch of training datasets
    # for each dataset:
    #   (hyper-)train good GRAFFs
    #   (hyper-)train good GCNs
    #   test GRAFFs and GCNs


if __name__ == '__main__':
    main()
