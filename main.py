from ads import data
from ads.models import GNN
from torch import nn

import itertools
import multiprocessing as mp
import numpy as np
import torch
import torch.optim


def test(net, dataloader, loss_fn, epoch, accuracy_score, logger=print):
    net.eval()

    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for datapoint in dataloader:
            pred = net(datapoint.x)
            total_loss += loss_fn(pred, datapoint.y).item()
            total_acc += accuracy_score(pred, datapoint.y).item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    logger(f'Validation set: Average loss: {avg_loss:.4f}')
    logger(f'Validation set: Average Acc: {avg_acc:.4f}')
    logger('\n')

    return avg_loss, avg_acc


def train(model, train_dl, val_dl, loss_fn, num_epochs, optim_gen, accuracy_score, logger=print):
    optim = optim_gen(model.parameters())
    losses = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_iteration(model, train_dl, optim, loss_fn, epoch, accuracy_score, logger=logger)
        val_loss, val_acc = test(model, val_dl, loss_fn, epoch, accuracy_score, logger=logger)
        losses.append({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
    return losses


def train_iteration(net, dataloader, optim, loss_fn, epoch, accuracy_score, logger=print):
    net.train()

    total_loss = 0
    total_acc = 0
    for datapoint in dataloader:
        pred = net(datapoint.x)

        y = datapoint.y
        loss = loss_fn(pred, y)
        net.zero_grad()
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        total_acc += accuracy_score(pred, y).item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    logger(f'Training set: Average loss: {avg_loss:.4f}')
    logger(f'Training set: Average Acc: {avg_acc:.4f}')
    return total_loss, total_acc


def evaluate_gnn_cora(pred, true):
    y_hat = pred.data.max(1)[1]
    num_correct = y_hat.eq(true).sum()
    num_total = len(true)
    accuracy = 100.0 * (num_correct / num_total)
    return accuracy


def train_with_params(iterations, train_dl, val_dl, hidden_dim, T, step_size, num_epochs, lr):
    results = []
    for i in range(iterations):
        print(f'starting iteration {i:4d}', end='\r')
        model = GNN.for_dataset(train_dl.__iter__().__next__(), hidden_dim=hidden_dim, T=T, step_size=step_size)
        # model(dataset.x)
        metrics = train(model, train_dl, val_dl, nn.CrossEntropyLoss(), num_epochs, lambda p: torch.optim.Adam(p, lr=lr), evaluate_gnn_cora, logger=lambda s: None)
        results.append(metrics[-1])
    print()
    return results

class GridSearcher:
    def __init__(self, train_dl, val_dl, iterations, num_choices, shared_kwargs):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.iterations = iterations
        self.num_choices = num_choices
        self.shared_kwargs = shared_kwargs

    def search_cell(self, idx_params):
        idx, params = idx_params
        def describe_val_acc(metrics):
            accs = []
            for m in metrics:
                accs.append(m['val_acc'])
            accs_np = np.array(accs)
            return accs_np.mean(), accs_np.std()
        kwargs = dict(params)
        print(f'training with params {idx+1:3d}/{self.num_choices:3d} ({idx/self.num_choices * 100:4.1f}%): {kwargs}')
        metrics = train_with_params(self.iterations, self.train_dl, self.val_dl, **kwargs, **self.shared_kwargs)
        ava, std = describe_val_acc(metrics)
        print(f'average acc: {ava:.4f}±{std:.4f}')
        return (ava, std, None)


def grid_search(iterations, train_dl, val_dl, param_grid, **shared_kwargs):
    # param_grid: Dict[ param_name -> List[param] ]
    # list_of_params: List[List[ (param_name, param_value)] ] where param_name is the same for all elements of any given inner list.
    list_of_params = [[(k, p) for p in ps] for k, ps in param_grid.items()]
    num_choices = np.array([len(ps) for ps in list_of_params]).prod()

    max_val_acc = 0
    argmax_args = None
    argmax_std = None
    argmax_model = None
    p = mp.Pool(2)
    searcher = GridSearcher(train_dl, val_dl, iterations, num_choices, shared_kwargs)
    results = p.map(searcher.search_cell, enumerate(itertools.product(*list_of_params)))
    p.close()
    p.join()
    for ava, std, model in results:
        if ava > max_val_acc:
            max_val_acc = ava
            argmax_std = std
            argmax_model = model
    print(f'best parameter choice was {argmax_args}, achieving avg val acc={max_val_acc:.4f}±{argmax_std:.4f}')
    return argmax_args, max_val_acc, argmax_std


def main():
    train_dl, val_dl, test_dl = data.get_data(10, 40, 4, 4, seed=1045966)
    param_grid = {
        "T": [1, 2, 3, 4, 5],
        # "step_size": [2, 1, 0.5, 0.25, 0.125],
        # "T": [3],
        "step_size": [0.25],
    }
    grid_search(1, train_dl, val_dl, param_grid, num_epochs=100, lr=0.0026, hidden_dim=64)
    # metrics = train_with_params(1, dataset, 64, 3, 0.25, 100, 0.0026)


if __name__ == '__main__':
    main()
