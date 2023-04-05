import copy
import torch
import torch.nn as nn
import uuid

from ads.models import GNN


def test(net, dataloader, loss_fn, epoch, accuracy_score, logger=print):
    net.eval()

    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for datapoint in dataloader:
            pred = net(datapoint.x)
            total_loss += loss_fn(pred, datapoint.y).item()
            total_acc += accuracy_score(pred, datapoint.y)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    logger(f'Validation set: Average loss: {avg_loss:.4f}')
    logger(f'Validation set: Average Acc: {avg_acc:.4f}')
    logger('\n')

    return avg_loss, avg_acc


def train(model, train_dl, val_dl, loss_fn, num_epochs, optim_gen, accuracy_score, logger=print):
    optim = optim_gen(model.parameters())
    metrics = []
    max_acc = 0
    best_model = None
    best_result = None
    for epoch in range(num_epochs):
        train_loss, train_acc = train_iteration(model, train_dl, optim, loss_fn, epoch, accuracy_score, logger=logger)
        val_loss, val_acc = test(model, val_dl, loss_fn, epoch, accuracy_score, logger=logger)
        result = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        if val_acc > max_acc:
            best_model = copy.deepcopy(model)
            best_result = result
        metrics.append(result)
    path = f'models/model-{uuid.uuid4()}.pt'
    torch.save(best_model, path)
    return metrics, path, best_result


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
        total_acc += accuracy_score(pred, y)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    logger(f'Training set: Average loss: {avg_loss:.4f}')
    logger(f'Training set: Average Acc: {avg_acc:.4f}')
    return avg_acc, avg_loss


def dice_acc(pred, true):
    eps = 1e-10
    y_hat = pred.data.max(1)[1]
    intersection = ((y_hat == 1) & (true == 1)).sum()
    union = (y_hat == 1).sum() + (true == 1).sum()
    return (2 * intersection + eps) / (union + eps)


def train_with_params(iterations, train_dl, val_dl, model_cls, hidden_dim, T, step_size, num_epochs, lr):
    all_metrics = []
    best_models = []
    best_results = []
    for i in range(iterations):
        print(f'starting iteration {i:4d}', end='\r')
        model = model_cls.for_dataset(train_dl.__iter__().__next__(), hidden_dim=hidden_dim, T=T, step_size=step_size)
        # model(dataset.x)
        model_metrics, best_model, best_result = train(model, train_dl, val_dl, nn.CrossEntropyLoss(), num_epochs, lambda p: torch.optim.Adam(p, lr=lr), dice_acc, logger=lambda s: None)
        all_metrics.append(model_metrics)
        best_models.append(best_model)
        best_results.append(best_result)
    print()
    return all_metrics, best_models, best_results
