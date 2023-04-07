import matplotlib.pyplot as plt
import torch

from ads.data import draw_graph, mk_datapoint, add_colors
from matplotlib.transforms import ScaledTranslation


def make_task_example(axes):
    seed = 1045967
    data = mk_datapoint(10, 4, 3, out_degree=2, seed=seed)

    add_colors(data, 4, 1, seed)
    pos = draw_graph(data, seed=seed, axis=axes[0])

    add_colors(data, 4, 3, seed)
    draw_graph(data, pos, seed, axis=axes[1])


def make_acc_vs_complexity(fig, ax):
    def load_data(files):
        dice = []
        std = []
        for file in files:
            obj = torch.load('results/' + file)
            dice.append(obj['testset_results']['avg_dice'])
            std.append(obj['testset_results']['std_dice'])
        return dice, std

    def load_benchmark_dice(files):
        dice = []
        for file in files:
            obj = torch.load('results/' + file)
            ratio = obj['ratio']
            dice.append(ratio / (1 + ratio))
        return dice

    DP_GRAFF = [
        "1680826914-0-results-graff.pt",
        "1680829240-1-results-graff.pt",
        "1680831600-2-results-graff.pt",
        "1680833923-3-results-graff.pt",
        "1680836256-4-results-graff.pt",
        "1680838577-5-results-graff.pt",
    ]
    DP_MULTI_GRAFF = [
        "1680828768-0-results-layered-graff.pt",
        "1680831125-1-results-layered-graff.pt",
        "1680833463-2-results-layered-graff.pt",
        "1680835791-3-results-layered-graff.pt",
        "1680838114-4-results-layered-graff.pt",
        "1680840432-5-results-layered-graff.pt",
    ]
    DP_GCN = [
        "1680829001-0-results-gcn.pt",
        "1680831363-1-results-gcn.pt",
        "1680833695-2-results-gcn.pt",
        "1680836022-3-results-gcn.pt",
        "1680838348-4-results-gcn.pt",
        "1680840666-5-results-gcn.pt",
    ]
    CLASS_SIZES = [
        "1680829001-0-class_sizes.pt",
        "1680831363-1-class_sizes.pt",
        "1680833695-2-class_sizes.pt",
        "1680836022-3-class_sizes.pt",
        "1680838348-4-class_sizes.pt",
        "1680840666-5-class_sizes.pt",
    ]

    complexities = list(range(1, 6 + 1))

    dice_graff, dice_graff_std = load_data(DP_GRAFF)
    t1 = ax.transData + ScaledTranslation(-5 / 72, 0, fig.dpi_scale_trans)
    ax.errorbar(complexities, dice_graff, yerr=dice_graff_std, label='GRAFF', transform=t1, capsize=5)

    dice_multi_graff, dice_multi_graff_std = load_data(DP_MULTI_GRAFF)
    t2 = ax.transData + ScaledTranslation(5 / 72, 0, fig.dpi_scale_trans)
    ax.errorbar(complexities, dice_multi_graff, yerr=dice_multi_graff_std, label='MultiGRAFF', transform=t2, capsize=5)

    dice_gcn, dice_gcn_std = load_data(DP_GCN)
    ax.errorbar(complexities, dice_gcn, yerr=dice_gcn_std, label='GCN', capsize=5)

    benchmark = load_benchmark_dice(CLASS_SIZES)
    ax.plot(complexities, benchmark, label='benchmark')

    ax.legend()


def main():
    fig = plt.figure(figsize=(8, 4))
    ex1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1, fig=fig)
    ex2 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1, fig=fig)
    a_vs_c = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=2, fig=fig)
    make_task_example([ex1, ex2])

    make_acc_vs_complexity(fig, a_vs_c)

    plt.savefig('img/the_figure.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
