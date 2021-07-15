import matplotlib

from matplotlib import pyplot as plt
import numpy as np
from utils import orth_check, map_to
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import torch
from mpl_toolkits.mplot3d import art3d
from matplotlib.colors import ListedColormap


def plot_advs(advs, orig=None, classes=None, orig_class=None, n=10,vmin=0,vmax=1):
    if orig is None:
        j = 0
    else:
        j = 1
    with_classes = True

    if classes is None:
        with_classes = False

    n = np.minimum(n, len(advs))
    dirs = advs - np.reshape(orig, (1,-1))

    max_val = np.maximum(abs(np.min(dirs)), abs(np.max(dirs)))
    min_val = - max_val

    dirs = np.reshape(dirs, [-1,28,28])
    advs = np.reshape(advs, [-1,28,28])
    fig, ax = plt.subplots(2, n + j, squeeze=False)

    if not (orig is None):
        orig = np.reshape(orig, [28, 28])
        ax[0, 0].set_title('original')
        ax[0, 0].imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[0, 0].set_xlabel(str(orig_class), fontdict={'fontsize': 15})

        ax[1, 0].axis('off')

    for i, (a, d) in enumerate(zip(advs[:n], dirs[:n])):
        ax[0, i + j].set_title('Adversarial ' + str(i + 1))
        if with_classes:
            ax[0, i + j].set_xlabel('\u279E ' + str(int(classes[i])), fontdict={'fontsize': 15})
        im_adv = ax[0, i + j].imshow(a.reshape([28, 28]), cmap='gray', vmin=vmin, vmax=vmax)
        ax[0, i + j].set_xticks([])
        ax[0, i + j].set_yticks([])
        ax[1, i + j].set_title('Perturbation ' + str(i + 1))

        im_pert = ax[1, i + j].imshow(d.reshape([28, 28]), vmin=min_val, vmax=max_val)
        ax[1, i + j].set_xticks([])
        ax[1, i + j].set_yticks([])

    #ax[1, n+j-1].set_xlabel('magnification factor ' + str(np.round(1/max_val, 2)), horizontalalignment='right', x=1.0)
    fig.colorbar(im_adv, ax=ax[0, :].ravel().tolist(), shrink=0.7)
    fig.colorbar(im_pert, ax=ax[1, :].ravel().tolist(), shrink=0.7)

    return fig, ax


def plot_mean_advs(advs, images, classes, labels, pert_lengths, n=10, vmin=0, vmax=1):

    mean_pert_length = np.mean(pert_lengths, axis=0)
    dist_to_mean = np.sum(np.abs(pert_lengths - mean_pert_length), axis=-1)
    min_idx = np.argmin(dist_to_mean)
    return plot_advs(advs[min_idx], images[min_idx], classes[min_idx], labels[min_idx], n=n, vmin=vmin, vmax=vmax)


def plot_pert_len_difs(advs_natural, advs_robust, images, n=10, ord=2):
    n = np.minimum(n, advs_natural[0].shape[-2])
    pert_len_natural = np.linalg.norm(advs_natural - images.reshape((-1, 1, 784)), ord=ord, axis=-1)
    pert_len_robust = np.linalg.norm(advs_robust - images.reshape((-1, 1, 784)), ord=ord, axis=-1)
    pert_len_natural[np.all(advs_natural == 0, axis=-1)] = np.nan
    pert_len_robust[np.all(advs_robust == 0, axis=-1)] = np.nan

    pert_len_difs = pert_len_robust - pert_len_natural
    pert_len_difs = pert_len_difs[:, :n]
    mask = ~np.isnan(pert_len_difs)
    filtered_data = [d[m] for d, m in zip(pert_len_difs.T, mask.T)]

    boxprops = dict(color='tab:blue', linewidth=1.5, alpha=0.7)
    whiskerprops = dict(color='tab:blue', alpha=0.7)
    capprops = dict(color='tab:blue', alpha=0.7)
    medianprops = dict(linestyle=None, linewidth=0)
    meanpointprops = dict(marker='o', markeredgecolor='black',
                          markerfacecolor='tab:orange')
    plt.boxplot(filtered_data, whis=[10, 90], showfliers=False, showmeans=True, boxprops=boxprops,
                whiskerprops=whiskerprops, capprops=capprops, meanprops=meanpointprops, medianprops=medianprops)
    plt.title('Difference of perturbation lengths between robust and natural model')
    plt.xlabel('d')
    if ord == np.inf:
        plt.ylabel('adversarial vector length ($\ell_\infty-norm$)')
    else:
        plt.ylabel('adversarial vector length ($\ell_%d-norm$)' % (ord))

    return fig, ax


def plot_pert_lengths(pert_lengths, n=10, labels=None, ord=2):
    n = np.minimum(n, pert_lengths[0].shape[1])
    pert_lengths = [p[:,:n] for p in pert_lengths]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    l = []

    fig, ax = plt.subplots()
    for i, pl in enumerate(pert_lengths):
        boxprops = dict(color=colors[i], linewidth=1.5, alpha=0.7)
        whiskerprops = dict(color=colors[i], alpha=0.7)
        capprops = dict(color=colors[i], alpha=0.7)
        medianprops = dict(linestyle=None, linewidth=0)
        meanpointprops = dict(marker='o', markeredgecolor='black',
                              markerfacecolor=colors[i])

        if not labels is None:
            l.append(mpatches.Patch(color=colors[i], label=labels[i]))

        pl[pl==0] = np.nan
        mask = ~np.isnan(pl)
        filtered_data = [d[m] for d, m in zip(pl.T, mask.T)]
        ax.boxplot(filtered_data, whis=[10,90], showfliers=False, showmeans=True, boxprops=boxprops,
                    whiskerprops=whiskerprops, capprops=capprops, meanprops=meanpointprops, medianprops=medianprops)
        ax.plot(range(1, n+1), np.nanmean(pl, axis=0), '--', color=colors[i])
    plt.title('Perturbation length of first ' + str(n) + ' adversarial directions')
    plt.xlabel('d')
    if ord == np.inf:
        plt.ylabel('adversarial vector length ($\ell_\infty-norm$)')
    else:
        plt.ylabel('adversarial vector length ($\ell_%d-norm$)' % (ord))
    if not (labels is None):
        plt.legend(handles=l)
    return fig, ax


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


def plot_cw_surface(orig, adv1, adv2, model):
    orig = np.reshape(orig, (784))
    len1 = np.linalg.norm(adv1-orig)
    len2 = np.linalg.norm(adv2-orig)
    dir1 = (adv1 - orig) / len1
    dir2 = (adv2 - orig) / len2

    n_grid = 100
    len_grid = 2.5
    offset = 0.1
    x = np.linspace(-offset, len_grid, n_grid)
    y = np.linspace(-offset, len_grid, n_grid)
    X, Y = np.meshgrid(x, y)
    advs = orig + (dir1*np.reshape(X,(-1,1)) + dir2*np.reshape(Y,(-1,1)))
    advs = np.array(np.reshape(advs, (-1,1,28,28)).astype('float64'), dtype='float32')
    input = torch.split(torch.tensor(advs),20)

    preds = np.empty((0,10))
    for batch in input:
        preds = np.concatenate((preds, model(batch).detach().cpu().numpy()),axis=0)
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=-1)[:, np.newaxis]
    orig_pred = model(torch.tensor(np.reshape(orig, (1, 1, 28, 28)))).detach().cpu().numpy()
    orig_pred = np.exp(orig_pred) / np.sum(np.exp(orig_pred), axis=-1)[:, np.newaxis]

    label = np.argmax(orig_pred)
    conf = preds[:,label].reshape((n_grid,n_grid))
    classes = np.argmax(preds,axis=-1).reshape((n_grid,n_grid))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plot_colors = np.empty(X.shape, dtype=object)
    colors = ['tab:orange', 'tab:green', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:blue','tab:cyan', 'tab:olive', 'tab:red', 'tab:purple']
    labels = []
    for i, c in enumerate(np.unique(classes)):
        labels.append(mpatches.Patch(color=colors[c], label='Class ' + str(c)))
        plot_colors[classes == c] = colors[c]

    # Plot the surface.
    ax.plot_surface(X, Y, conf, linewidth=0, antialiased=False, facecolors=plot_colors)
    p = mpatches.Circle((offset, offset), .1, ec='k', fc='k')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=orig_pred[0, label], zdir='z')


    ax.set_xlabel('dir 1 ($\ell_2-length$)')
    ax.set_ylabel('dir 2 ($\ell_2-length$)')
    ax.set_zlabel('confidence in original class')
    ax.set_zlim((0,1))

    # Add legend with proxy artists
    plt.legend(handles=labels, title='class with largest confidence')
    plt.show()


def plot_dec_space(orig, adv1, adv2, model, show_legend=True, show_advs=True):
    orig = np.reshape(orig, (784))
    len1 = np.linalg.norm(adv1-orig)
    len2 = np.linalg.norm(adv2-orig)
    dir1 = (adv1 - orig) / len1
    dir2 = (adv2 - orig) / len2

    n_grid = 100
    len_grid = np.maximum(2.5, np.maximum(len1,len2)+1)
    offset = 0.1
    x = np.linspace(-offset, len_grid, n_grid)
    y = np.linspace(-offset, len_grid, n_grid)
    X, Y = np.meshgrid(x, y)
    advs = orig + (dir1 * np.reshape(X, (-1, 1)) + dir2 * np.reshape(Y, (-1, 1)))
    advs = np.array(np.reshape(advs, (-1, 1, 28, 28)).astype('float64'), dtype='float32')
    input = torch.split(torch.tensor(advs), 20)

    preds = np.empty((0, 10))
    for batch in input:
        preds = np.concatenate((preds, model(batch).detach().cpu().numpy()), axis=0)

    classes = np.argmax(preds, axis=-1).reshape((n_grid, n_grid))

    ax = plt.gca()
    colors = ['tab:orange', 'tab:green', 'tab:brown', 'tab:grey', 'tab:blue', 'tab:pink','tab:cyan', 'tab:olive', 'tab:red', 'tab:purple']
    labels = []
    colorList = []
    for i, c in enumerate(np.unique(classes)):
        labels.append(mpatches.Patch(color=colors[c], label=str(c)))
        colorList.append(colors[c])
    # Plot the surface.
    new_cmap = ListedColormap(colors)

    plt.imshow(classes, cmap=new_cmap, origin='lower', vmin=0, vmax=9)
    plt.axvline(offset*n_grid/(offset+len_grid), c='k', ls='--', alpha=0.5)
    plt.axhline(offset*n_grid/(offset+len_grid), c='k', ls='--', alpha=0.5)

    plt.plot(offset*n_grid/(offset+len_grid), offset*n_grid/(offset+len_grid),
             markeredgecolor='black', markerfacecolor='black', marker='o')

    if show_advs:
        plt.plot((offset+len1)*n_grid/(offset+len_grid), offset*n_grid/(offset+len_grid),
                 markeredgecolor='black', markerfacecolor='tab:red', marker='o')
        plt.plot(offset*n_grid/(offset+len_grid), (offset+len2)*n_grid/(offset+len_grid),
                 markeredgecolor='black', markerfacecolor='tab:red', marker='o')

    ax.set_xlabel('dir 1 ($\ell_2$-length)', fontdict={'fontsize': 15})
    ax.set_ylabel('dir 2 ($\ell_2$-length)', fontdict={'fontsize': 15})

    plt.xticks(np.linspace(offset*n_grid/(offset+len_grid), (offset+np.floor(len_grid))*n_grid/(offset+len_grid), 5),
               [np.round(x, 2).astype(str) for x in np.linspace(0, np.floor(len_grid), 5)])
    plt.yticks(np.linspace(offset*n_grid/(offset+len_grid), (offset+np.floor(len_grid))*n_grid/(offset+len_grid), 5),
               [np.round(x, 2).astype(str) for x in np.linspace(0, np.floor(len_grid), 5)])

    if show_legend:
        # Add legend with proxy artists
        plt.legend(handles=labels, title='predicted class')


def plot_contrasted_dec_space(orig, adv1, adv2, model, n=4):
    epsilon = np.linspace(0, 0.5, n+2)
    dir1 = adv1 - np.reshape(orig, (784))
    dir2 = adv2 - np.reshape(orig, (784))

    n_cols = int(np.ceil(n+2)/2)
    n_rows = int(np.ceil((n+2)/n_cols))
    fig, ax = plt.subplots(2, n+2)

    for i in range(n+2):
        scaled_orig = map_to(orig, epsilon[i], 1-epsilon[i])

        plt.subplot(2, n+2, i+1)
        plot_dec_space(scaled_orig, dir1+np.reshape(scaled_orig, (784)), dir2+np.reshape(scaled_orig, (784)), model, show_legend=False)
        plt.title('contrast = %.1f' % (1-2*epsilon[i]))
        # if i < (n+2-n_cols):
        #     plt.gca().xaxis.set_visible(False)
        if not (i % n_cols) == 0:
            plt.gca().yaxis.set_visible(False)

        plt.subplot(2, n+2, i+1+n+2)
        plt.imshow(scaled_orig[0], cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])

    # colors = ['tab:orange', 'tab:green', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:blue', 'tab:cyan', 'tab:olive',
    #           'tab:red', 'tab:purple']
    # labels = []
    # for c in range(10):
    #     labels.append(mpatches.Patch(color=colors[c], label='Class ' + str(c)))
    # fig.legend(handles=labels, title='predicted class', loc='upper right', bbox_to_anchor=(0.95, 0.7))
    # plt.subplots_adjust(right=0.9)
    plt.show()


def plot_var_hist(classes, labels, title=None, with_colors=True):
    bar_width = 0.4
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:brown', 'tab:grey', 'tab:pink',
              'tab:cyan', 'tab:olive']
    data = np.zeros((10,10))
    for l in range(10):
        var = np.mean(np.array([len(np.unique(x[~np.isnan(x)])) for x in classes[labels == l]]))
        u, c = np.unique(classes[labels == l], return_counts=True)
        c = c[~np.isnan(u)]
        u = u[~np.isnan(u)]
        data[u.astype(int), l] = c / np.sum(c) * var

    y_off = np.zeros(10)
    for idx in range(10):
        if with_colors:
            plt.bar(range(10), data[idx], bar_width, bottom=y_off,  color=colors[idx])
        else:
            plt.bar(range(10), data[idx], bar_width, bottom=y_off, color='k')
        y_off += data[idx]
    plt.xlabel('original class label')
    plt.ylabel('mean number of adversarial classes')
    plt.legend(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'], title='adversarial label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(10))
    if not title is None:
        plt.title(title)
    plt.ylim(0,2.5)
    plt.tight_layout()
    plt.show()


def show_consistency(adv_dirs):
    inner_prods = []
    for i in range(adv_dirs.shape[1]):
        all_zero = np.all(adv_dirs[:, i] == 0, axis=-1)
        orth = orth_check(adv_dirs[~all_zero, i])
        inner_prods.append(abs(orth[np.triu_indices_from(orth, 1)]))
    plt.figure()
    plt.boxplot(inner_prods, showfliers=False, whis='range')
    plt.xlabel('d')
    plt.ylabel('inner poduct across runs')
    # plt.title('Consistency of Adversarial Directions over %d runs' % (adv_dirs.shape[0]))
    plt.show()


def show_orth(adv_dirs):

    orth = orth_check(adv_dirs)
    table = plt.table(np.around(orth, decimals=2), loc='center')
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    plt.xticks([])
    plt.yticks([])
    plt.box()
    plt.title('Orthorgonality of adversarial directions', y=0.9)