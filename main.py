import itertools
import torch
import time

from ads import data
from ads.grid_search import grid_search
from ads.models import MultiGRAFFNetwork, GRAFFNetwork, ClassicGNN
from ads.training import score_results
from datetime import datetime


def class_sizes(dl):
    yes_class = 0
    all_classes = 0
    for d in dl:
        yes_class += d.y.sum()
        all_classes += len(d.y)
    return {
        "all": all_classes,
        "yes": yes_class,
        "ratio": yes_class / all_classes,
    }


def train_good_graff(iterations, num_epochs, train_dl, val_dl):
    param_grid = {
        # "T": [2, 3, 4],
        # "step_size": [1, 0.5, 0.25, 0.125],
        "T": [2, 3],
        # "step_size": [0.25, 0.5],
        # "dropout": [0.25, 0.5, 0.75]
        "hidden_dim": [4],
        "step_size": [0.25, 0.5],
        "dropout": [0.25]
    }
    return grid_search(iterations, train_dl, val_dl, GRAFFNetwork, param_grid, num_epochs=num_epochs, lr=0.0026)


def train_good_gcn(iterations, num_epochs, train_dl, val_dl):
    param_grid = {
        # "hidden_dim": [4, 8],
        "num_layers": [3, 4, 5, 6],
        # "lr": [0.01]
        "hidden_dim": [4],
        # "num_layers": [3],
        "lr": [0.01]
    }
    return grid_search(iterations, train_dl, val_dl, ClassicGNN, param_grid, num_epochs=num_epochs)


def train_good_multi_graff(iterations, num_epochs, train_dl, val_dl):
    param_grid = {
        "hidden_dim": [4],
        "T": [2, 3],
        "step_size": [0.5, 1],
        "num_layers": [2, 4],
        "ev_trans_shared": [True, False]
    }
    return grid_search(iterations, train_dl, val_dl, MultiGRAFFNetwork, param_grid, num_epochs=num_epochs, lr=0.01, dropout=0)


def main():
    start_time = datetime.now()
    seed = itertools.count(1045966)
    iterations = 5
    graph_size = 100
    num_datapoints = 50
    num_epochs = 100
    datasets = [
        data.get_data(num_datapoints, graph_size, 6, 1, seed=seed.__next__()),
        data.get_data(num_datapoints, graph_size, 6, 2, seed=seed.__next__()),
        data.get_data(num_datapoints, graph_size, 6, 3, seed=seed.__next__()),
        data.get_data(num_datapoints, graph_size, 6, 4, seed=seed.__next__()),
        data.get_data(num_datapoints, graph_size, 6, 5, seed=seed.__next__()),
        data.get_data(num_datapoints, graph_size, 6, 6, seed=seed.__next__()),
    ]

    for idx, (train_dl, val_dl, test_dl) in enumerate(datasets):
        if True:
            r = train_good_graff(iterations, num_epochs, train_dl, val_dl)
            print(f'GRAFF: best parameter choice was {r["args"]}, achieving avg val acc={r["acc"]:.4f}±{r["std"]:.4f}')
            r['testset_results'] = score_results(test_dl, r)
            print(f'testset results: {r["testset_results"]}')
            torch.save(r, f'results/{int(time.time())}-{idx}-results-graff.pt')
            print('\n')

        if True:
            r = train_good_multi_graff(iterations, num_epochs, train_dl, val_dl)
            print(f'MultiGRAFFNetwork: best parameter choice was {r["args"]}, achieving avg val acc={r["acc"]:.4f}±{r["std"]:.4f}')
            r['testset_results'] = score_results(test_dl, r)
            print(f'testset results: {r["testset_results"]}')
            torch.save(r, f'results/{int(time.time())}-{idx}-results-layered-graff.pt')
            print('\n')

        if True:
            r = train_good_gcn(iterations, num_epochs, train_dl, val_dl)
            print(f'GCN: best parameter choice was {r["args"]}, achieving avg val acc={r["acc"]:.4f}±{r["std"]:.4f}')
            r['testset_results'] = score_results(test_dl, r)
            print(f'testset results: {r["testset_results"]}')
            torch.save(r, f'results/{int(time.time())}-{idx}-results-gcn.pt')
            print('\n')

        torch.save(class_sizes(test_dl), f'results/{int(time.time())}-{idx}-class_sizes.pt')
    end_time = datetime.now()
    print(end_time)
    print(end_time - start_time)

    # Generate a bunch of training datasets
    # for each dataset:
    #   (hyper-)train good GRAFFs
    #   (hyper-)train good GCNs
    #   test GRAFFs and GCNs


if __name__ == '__main__':
    main()
