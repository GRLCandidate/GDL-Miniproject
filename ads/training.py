import copy
import numpy as np
import torch
import torch.nn as nn
import uuid

from ads.models import DiceLoss


def test(net, dataloader, loss_fn, epoch, accuracy_score, logger=print):
    net.eval()

    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for datapoint in dataloader:
            pred = net(datapoint)
            total_loss += loss_fn(pred, datapoint.y).item()
            total_acc += accuracy_score(pred, datapoint.y)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    logger(f'Validation set: Average loss: {avg_loss:.4f}')
    logger(f'Validation set: Average Acc: {avg_acc:.4f}')
    logger('\n')

    return avg_loss, avg_acc


def train(model, train_dl, val_dl, loss_fn, num_epochs, optim_gen, accuracy_score, logger=print):
    if logger is None:
        logger = lambda x: x
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
            max_acc = val_acc
            best_model = copy.deepcopy(model)
            best_result = result
        metrics.append(result)
    path = f'models/model-{uuid.uuid4()}.pt'
    torch.save(best_model, path)
    # print(f'best result: {best_result}')
    return metrics, path, best_result


def train_iteration(net, dataloader, optim, loss_fn, epoch, accuracy_score, logger=print):
    net.train()

    total_loss = 0
    total_acc = 0
    for idx, datapoint in enumerate(dataloader):
        pred = net(datapoint)

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
    logger(y[:10])
    logger(f'Training set: Average loss: {avg_loss:.4f}')
    logger(f'Training set: Average Acc: {avg_acc:.4f}')
    return avg_loss, avg_acc


def dice_acc(pred, true):
    eps = 1e-10
    y_hat = torch.argmax(pred, dim=1)
    intersection = ((y_hat == 1) & (true == 1)).sum()
    union = (y_hat == 1).sum() + (true == 1).sum()
    return (2 * intersection + eps) / (union + eps)


def acc(pred, true):
    y_hat = torch.argmax(pred, dim=1)
    correct = (y_hat == true).sum().item()
    return correct / len(true)


def train_with_params(iterations, train_dl, val_dl, model_cls, num_epochs, lr, **kwargs):
    all_metrics = []
    best_models = []
    best_results = []
    for i in range(iterations):
        print(f'starting iteration {i:4d}', end='\r')
        model = model_cls.for_dataset(train_dl.__iter__().__next__(), **kwargs)
        # model(dataset.x)
        model_metrics, best_model, best_result = train(model, train_dl, val_dl, DiceLoss(), num_epochs, lambda p: torch.optim.Adam(p, lr=lr), dice_acc, logger=None)
        all_metrics.append(model_metrics)
        best_models.append(best_model)
        best_results.append(best_result)
    print()
    return all_metrics, best_models, best_results


def score_results(test_dl, r):
    models = r['models']
    dice_scores = []
    accuracies = []

    for model in models:
        model_dice = 0
        model_acc = 0
        for data in test_dl:
            pred = model(data)
            model_dice += dice_acc(pred, data.y)
            model_acc += acc(pred, data.y)
        dice_scores.append(model_dice / len(test_dl))
        accuracies.append(model_acc / len(test_dl))

    dice_scores_np = np.array(dice_scores)
    accuracies_np = np.array(accuracies)
    return {
        'avg_acc': accuracies_np.mean(),
        'std_acc': accuracies_np.std(),
        'avg_dice': dice_scores_np.mean(),
        'std_dice': dice_scores_np.std(),
    }
