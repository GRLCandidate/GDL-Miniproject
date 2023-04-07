import numpy as np
import itertools
import multiprocessing as mp
import torch

from ads.training import train_with_params


class GridSearcher:
    def __init__(self, train_dl, val_dl, model_cls, iterations, num_choices, shared_kwargs):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.iterations = iterations
        self.num_choices = num_choices
        self.shared_kwargs = shared_kwargs
        self.model_cls = model_cls

    def search_cell(self, idx_params):
        def describe_val_acc(metrics):
            accs = []
            for m in metrics:
                accs.append(m['val_acc'])
            accs_np = np.array(accs)
            return accs_np.mean(), accs_np.std()
        idx, params = idx_params
        kwargs = dict(params)
        # print(f'training with params {idx+1:3d}/{self.num_choices:3d} ({idx/self.num_choices * 100:4.1f}%): {kwargs}')
        metrics, models, results = train_with_params(self.iterations, self.train_dl, self.val_dl, self.model_cls, **kwargs, **self.shared_kwargs)
        ava, std = describe_val_acc(results)
        # print(f'average acc: {ava:.4f}±{std:.4f}')
        # print()
        return {
            'acc': ava,
            'metrics': metrics,
            'models': models,
            'params': params,
            'std': std,
        }


def grid_search(iterations, train_dl, val_dl, model_cls, param_grid, comment=None, **shared_kwargs):
    # param_grid: Dict[ param_name -> List[param] ]
    # list_of_params: List[List[ (param_name, param_value)] ] where param_name is the same for all elements of any given inner list.
    list_of_params = [[(k, p) for p in ps] for k, ps in param_grid.items()]
    num_choices = np.array([len(ps) for ps in list_of_params]).prod()

    max_val_acc = 0
    argmax_args = None
    argmax_std = None
    argmax_model_paths = None
    argmax_metrics = None
    p = mp.Pool(2)
    searcher = GridSearcher(train_dl, val_dl, model_cls, iterations, num_choices, shared_kwargs)
    # results = p.map(searcher.search_cell, enumerate(itertools.product(*list_of_params)))
    results = map(searcher.search_cell, enumerate(itertools.product(*list_of_params)))
    p.close()
    p.join()
    for result in results:
        ava = result['acc']
        if ava > max_val_acc:
            max_val_acc = ava
            argmax_std = result['std']
            argmax_model_paths = result['models']
            argmax_args = result['params']
            argmax_metrics = result['metrics']
    argmax_models = [torch.load(path) for path in argmax_model_paths]
    ret = {
        'param_grid': param_grid,
        'shared_kwargs': shared_kwargs,
        'dl': train_dl.description,
        'comment': comment,
        'args': argmax_args,
        'acc': max_val_acc,
        'std': argmax_std,
        'models': argmax_models,
        'metrics': argmax_metrics,
        'all_results': results,
    }
    return ret
