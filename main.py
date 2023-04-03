from ads import data
from ads.models import GNN
from torch import nn

import itertools
import numpy as np
import torch
import torch.optim


def test(net, dataset, mask, loss_fn, epoch, accuracy_score, logger=print):
    net.eval()

    with torch.no_grad():
        pred = net(dataset.x)[mask]
        loss = loss_fn(pred, dataset.y[mask]).item()
        acc = accuracy_score(pred, dataset.y[mask]).item()

    logger(f'Validation set: Average loss: {loss:.4f}')
    logger(f'Validation set: Average Acc: {acc:.4f}')
    logger('\n')

    return loss, acc


def train(model, dataset, loss_fn, num_epochs, optim_gen, accuracy_score, logger=print):
    optim = optim_gen(model.parameters())
    losses = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_iteration(model, dataset, optim, loss_fn, epoch, accuracy_score, logger=logger)
        val_loss, val_acc = test(model, dataset, dataset.val_mask, loss_fn, epoch, accuracy_score, logger=logger)
        losses.append({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
    return losses


def train_iteration(net, dataset, optim, loss_fn, epoch, accuracy_score, logger=print):
    net.train()

    pred = net(dataset.x)[dataset.train_mask]

    #print(f'pred: {pred.shape}, target: {target.shape}')
    y = dataset.y[dataset.train_mask]
    loss = loss_fn(pred, y)
    net.zero_grad()
    optim.zero_grad()
    loss.backward()
    optim.step()

    loss = loss.item()
    acc = accuracy_score(pred, y).item()

    logger(f'Training set: Average loss: {loss:.4f}')
    logger(f'Training set: Average Acc: {acc:.4f}')
    return loss, acc


def evaluate_gnn_cora(pred, true):
    y_hat = pred.data.max(1)[1]
    num_correct = y_hat.eq(true).sum()
    num_total = len(true)
    accuracy = 100.0 * (num_correct / num_total)
    return accuracy


def train_with_params(iterations, dataset, hidden_dim, T, step_size, num_epochs, lr):
    results = []
    for i in range(iterations):
        print(f'starting iteration {i:4d}', end='\r')
        model = GNN.for_dataset(dataset, hidden_dim=hidden_dim, T=T, step_size=step_size)
        # model(dataset.x)
        metrics = train(model, dataset, nn.CrossEntropyLoss(), num_epochs, lambda p: torch.optim.Adam(p, lr=lr), evaluate_gnn_cora, logger=lambda s: None)
        results.append(metrics[-1])
    print()
    return results


def grid_search(iterations, dataset, param_grid, **shared_kwargs):
    def describe_val_acc(metrics):
        accs = []
        for m in metrics:
            accs.append(m['val_acc'])
        accs_np = np.array(accs)
        return accs_np.mean(), accs_np.std()
    # param_grid: Dict[ param_name -> List[param] ]

    # list_of_params: List[List[ (param_name, param_value)] ] where param_name is the same for all elements of any given inner list.
    list_of_params = [[(k, p) for p in ps] for k, ps in param_grid.items()]
    num_choices = np.array([len(ps) for ps in list_of_params]).prod()

    max_val_acc = 0
    argmax_args = None
    argmax_std = None
    for idx, params in enumerate(itertools.product(*list_of_params)):
        kwargs = dict(params)
        print(f'training with params {idx+1:3d}/{num_choices:3d} ({idx/num_choices * 100:4.1f}%): {kwargs}')
        metrics = train_with_params(iterations, dataset, **kwargs, **shared_kwargs)
        ava, std = describe_val_acc(metrics)
        if ava > max_val_acc:
            max_val_acc = ava
            argmax_args = kwargs
            argmax_std = std
        print(f'average acc: {ava:.4f}±{std:.4f}')
    print(f'best parameter choice was {argmax_args}, achieving avg val acc={max_val_acc:.4f}±{argmax_std:.4f}')
    return argmax_args, max_val_acc, argmax_std


def main():
    dataset = data.get_data()
    param_grid = {
        "T": [1, 2, 3, 4, 5],
        "step_size": [2, 1, 0.5, 0.25, 0.125],
        # "T": [3],
        # step_size": [0.25],
    }
    grid_search(10, dataset, param_grid, num_epochs=100, lr=0.0026, hidden_dim=64)
    # metrics = train_with_params(1, dataset, 64, 3, 0.25, 100, 0.0026)


if __name__ == '__main__':
    main()
