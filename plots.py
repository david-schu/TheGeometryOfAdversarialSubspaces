import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from utils import orth_check
from matplotlib.ticker import FormatStrFormatter

def plot_advs(advs, orig=None, classes=None, orig_class=None, n=10,vmin=0,vmax=1):
    if orig is None:
        j = 0
    else:
        j = 1
    with_classes = True
    if classes is None:
        with_classes = False
    n = np.minimum(n, len(advs))
    advs = np.reshape(advs, [-1,28,28])
    fig, ax = plt.subplots(1, n + j, squeeze=False)

    if not (orig is None):
        orig = np.reshape(orig, [28, 28])
        ax[0, 0].set_title('original')
        ax[0, 0].imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])

    for i, a in enumerate(advs[:n]):
        ax[0, i + j].set_title('Adversarial ' + str(i + 1))
        if with_classes:
            ax[0, i + j].set_xlabel(str(orig_class) + ' \u279E ' + str(int(classes[i])))
        ax[0, i + j].imshow(a.reshape([28, 28]), cmap='gray', vmin=vmin, vmax=vmax)
        ax[0, i + j].set_xticks([])
        ax[0, i + j].set_yticks([])
    plt.show()
    return


def plot_dirs(dirs, n=10, vmin=0, vmax=1):
    n = np.minimum(n, len(dirs))
    dirs = np.reshape(dirs, [-1,28,28])
    fig, ax = plt.subplots(1, n, squeeze=False)

    for i, d in enumerate(dirs[:n]):
        ax[0, i].set_title('Perturbation ' + str(i + 1))
        ax[0, i].imshow(d.reshape([28, 28]), cmap='gray', vmin=vmin, vmax=vmax)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
    plt.suptitle('Perturbations with magnification factor %.2f' % (1/vmax))
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
    plt.title('Orthorgonality of adversarial directions', y=0.1)
    # plt.show()
    return


def plot_pert_lengths(adv_class,pert_lengths):
    plt.figure(figsize=(7, 5))
    classes = np.unique(adv_class)
    for c in classes:
        plt.scatter(np.argwhere(np.array(adv_class) == c)+1,
                    pert_lengths[np.array(adv_class) == c],
                    label='target class ' + str(c))
    plt.title('Perturbation lengths of first ' + str(len(pert_lengths)) + 'adversarial directions')
    plt.xlabel('n')
    plt.ylabel('adversarial vector length ($l2-norm$)')
    plt.legend()
    plt.show()
    return


def plot_losses(losses, orth_const):
    idx = np.argmin(losses)
    fig, ax = plt.subplots(2, 2, squeeze=False)
    fig.subplots_adjust(hspace=0.5)
    plt.suptitle('orth_const =' + str(orth_const))
    ax[0, 0].set_title('overall loss')
    ax[0, 0].plot(range(idx), losses[0, :idx])
    ax[0, 1].set_title('is_adversarial loss')
    ax[0, 1].plot(range(idx), losses[1, :idx])
    ax[1, 0].set_title('squared_norms loss')
    ax[1, 0].plot(range(idx), losses[2, :idx])
    ax[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1, 1].set_title('is_orth loss')
    ax[1, 1].plot(range(idx), losses[3, :idx])

    return fig, ax

