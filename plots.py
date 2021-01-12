import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from utils import orth_check
from matplotlib.ticker import FormatStrFormatter
import torch


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


def plot_pert_lengths(pert_lengths, n=5, labels=None):
    n = np.minimum(n, pert_lengths.shape[-1])

    for p in pert_lengths:
        p = p[:, :n]
        p[p == 0] = np.nan
        pert_lengths_mean = np.nanmean(p, axis=0)
        pert_lengths_std = np.nanstd(p, axis=0)
        plt.errorbar(np.arange(len(pert_lengths_mean))+1, pert_lengths_mean, pert_lengths_std, fmt='o', alpha=0.7)
    plt.title('Perturbation length of first ' + str(pert_lengths.shape[-1]) + ' adversarial directions')
    plt.xlabel('n')
    plt.xticks(np.arange(len(pert_lengths_mean))+1)
    plt.ylabel('adversarial vector length ($l2-norm$)')
    plt.ylim(0)
    if not (labels is None):
        plt.legend(labels)
    plt.show()
    return


def plot_pert_lengths_single(adv_class, pert_lengths):
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


def plot_label_heatmap(classes, labels, show_all=True):
    adv_table = np.zeros((10, 10))
    for i in range(10):
        u, counts = np.unique(classes[labels == i], return_counts=True)
        counts = counts[~np.isnan(u)]
        u = u[~np.isnan(u)]
        adv_table[i, u.astype(int)] = counts/sum(counts)

    if show_all:

        n = classes.shape[-1]+1
        cols = int(np.ceil(n/2))
        rows = int(np.ceil(n/cols))
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
        ax[0, 0].imshow(adv_table, vmin=0, vmax=1)
        ax[0, 0].set_yticks(range(10))
        ax[0, 0].set_xticks(range(10))
        ax[0, 0].set_title('All Adversarials')

        for j, a in enumerate(ax.flatten()[1:]):
            adv_table = np.zeros((10, 10))
            for i in range(10):
                u, counts = np.unique(classes[labels == i, j], return_counts=True)
                counts = counts[~np.isnan(u)]
                u = u[~np.isnan(u)]
                adv_table[i, u.astype(int)] = counts / sum(counts)
            im = a.imshow(adv_table, vmin=0, vmax=1)
            a.set_title('Adversarial ' + str(j+1))

        fig.text(0.5, 0.04, 'adversarial class', ha='center')
        fig.text(0.04, 0.5, 'original class', va='center', rotation='vertical')

        cbar = fig.colorbar(im, ax=ax.ravel().tolist())
        cbar.ax.set_ylabel('normalized frequency', rotation=-90, va="bottom")
        plt.suptitle('Frequencies of adversarial classes')

    else:
        ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(adv_table, vmin=0, vmax=1)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('normalized frequency', rotation=-90, va="bottom")
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        ax.set_xlabel('adversarial class')
        ax.set_ylabel('original class')
        ax[0, 0].set_yticks(range(10))
        ax[0, 0].set_xticks(range(10))
        ax.xaxis.set_label_position('top')
    plt.show()
    return


def plot_cw_surface(orig, adv1, adv2, model):
    orig = np.reshape(orig, (784))
    dir1 = adv1 - orig
    dir2 = adv2 - orig

    x = np.linspace(-2,2,100)
    y = np.linspace(-2,2,100)
    X, Y = np.meshgrid(x, y)
    advs = orig + (dir1*np.ravel(X)[:,np.newaxis] + dir2*np.ravel(Y)[:, np.newaxis])
    advs = np.array(np.reshape(advs, (-1,1,28,28)).astype('float64'),dtype='float32')

    preds = model(torch.tensor(advs)).detach().cpu().numpy()
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=-1)[:, np.newaxis]
    orig_pred = model(torch.tensor(np.reshape(orig, (1, 1, 28, 28)))).detach().cpu().numpy()
    label = np.argmax(orig_pred)

    conf = preds[:,label].reshape((100,100))
    classes = np.argmax(preds,axis=-1).reshape((100,100))



    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plot_colors = np.empty(X.shape, dtype=object)
    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:purple', 'tab:red']
    for i, c in enumerate(np.unique(classes)):
        plot_colors[classes==c] = colors[i]
    # Plot the surface.
    ax.plot_surface(X, Y, conf, linewidth=0, antialiased=False, facecolors=plot_colors)


    plt.show()
    return

