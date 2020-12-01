import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from utils import orth_check


def plot_advs(orig, advs, n):
    n = np.minimum(n,len(advs)) + 1
    fig, ax = plt.subplots(1, n, squeeze=False)
    ax[0, 0].set_title('original')
    ax[0, 0].imshow(orig, cmap='gray', vmin=0, vmax=1)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    for i, a in enumerate(advs[:n]):
        ax[0, i + 1].set_title('Adversarial ' + str(i + 1))
        ax[0, i + 1].imshow(a.reshape([28, 28]), cmap='gray', vmin=0, vmax=1)
        ax[0, i + 1].set_xticks([])
        ax[0, i + 1].set_yticks([])
    plt.show()
    return


def show_orth(adv_dirs):
    orth = orth_check(adv_dirs)
    plt.figure()
    table = plt.table(np.around(orth, decimals=2), loc='center')
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    plt.xticks([])
    plt.yticks([])
    plt.box()
    plt.title('Orthorgonality of adversarial directions')
    plt.show()
    return


def plot_pert_lengths(adv_class,pert_lengths):
    plt.figure(figsize=(7, 5))
    classes = np.unique(adv_class)
    for c in classes:
        plt.scatter(np.argwhere(np.array(adv_class) == c),
                    pert_lengths[np.array(adv_class) == c],
                    label='target class ' + str(c))
    plt.xlabel('n')
    plt.ylabel('adversarial vector length ($l_2$)')
    plt.legend()
    plt.show()
    return


def plot_losses(losses,):
    idx = np.argmin(losses)

    fig, ax = plt.subplots(2, 2, squeeze=False)
    fig.subplots_adjust(hspace=0.5)
    ax[0, 0].set_title('loss')
    ax[0, 0].plot(range(idx), losses[0, :idx])
    ax[0, 1].set_title('is_adversarial_loss')
    ax[0, 1].plot(range(idx), losses[1, :idx])
    ax[1, 0].set_title('squared_norms')
    ax[1, 0].plot(range(idx), losses[2, :idx])
    ax[1, 1].set_title('is_orth')
    ax[1, 1].plot(range(idx), losses[3, :idx])
    return fig, ax

