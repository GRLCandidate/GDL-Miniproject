import matplotlib.pyplot as plt

from ads.data import draw_graph, mk_datapoint, add_colors


def make_task_example():
    seed = 1045967
    data = mk_datapoint(20, 4, 3, out_degree=2, seed=seed)

    fix, axis = plt.subplots(1, 2)

    add_colors(data, 4, 1, seed)
    pos = draw_graph(data, seed=seed, axis=axis[0])

    add_colors(data, 4, 3, seed)
    draw_graph(data, pos, seed, axis=axis[1])
    plt.savefig("img/task_example.pdf")


def main():
    make_task_example()


if __name__ == '__main__':
    main()
