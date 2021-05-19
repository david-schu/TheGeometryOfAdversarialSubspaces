import matplotlib
import plots as pl
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches
import torch
import time

from utils import map_to


def two_origs_movie(orig1, orig2, adv1, adv2, model, save_folder, n=100):
    matplotlib.use('Agg')
    orig1 = orig1.reshape([28, 28])
    orig2 = orig2.reshape([28, 28])
    epsilon = np.linspace(0, 1, n)
    dir1 = adv1 - np.reshape(orig1, (784))
    dir2 = adv2 - np.reshape(orig1, (784))

    diff = orig2 - orig1
    path = '../data/' + save_folder
    if not os.path.isdir(path):
        os.mkdir(path)

    for i in range(n):
        if i == 0:
            show_advs = True
        else:
            show_advs = False
        scaled_orig = orig1 + epsilon[i]*diff
        f, ax = plt.subplots(nrows=2, figsize=(8.4, 12))
        plt.subplot(2, 1, 1)
        pl.plot_dec_space(scaled_orig, dir1 + np.reshape(scaled_orig, (784)), dir2 + np.reshape(scaled_orig, (784)),
                          model,
                          show_legend=False, show_advs=show_advs)

        plt.subplot(2, 1, 2)
        plt.imshow(scaled_orig, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])

        colors = ['tab:orange', 'tab:green', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:blue', 'tab:cyan', 'tab:olive',
                  'tab:red', 'tab:purple']
        labels = []
        for c in range(10):
            labels.append(mpatches.Patch(color=colors[c], label=str(c)))
        lg = f.legend(handles=labels, title='predicted class', loc='upper right', bbox_to_anchor=(0.95, 0.85),
                      prop={'size': 14})
        lg.get_title().set_fontsize(15)
        plt.subplots_adjust(right=0.9, left=0.02)

        plt.savefig(path + '/%03d.png' % (i))
        plt.close()

def contrast_movie(orig, adv1, adv2, model, n=100, k=0):
    matplotlib.use('Agg')

    orig = orig.reshape([28, 28])
    epsilon = np.linspace(0, 0.5, n)
    dir1 = adv1 - np.reshape(orig, (784))
    dir2 = adv2 - np.reshape(orig, (784))
    if not os.path.isdir('../data/dec_movie%d/' % (k)):
        os.mkdir('../data/dec_movie%d/' % (k))

    for i in range(n):
        if i == 0:
            show_advs = True
        else:
            show_advs = False
        scaled_orig = map_to(orig, epsilon[i], 1 - epsilon[i])

        f, ax = plt.subplots(nrows=2, figsize=(8.4, 12))
        plt.subplot(2, 1, 1)
        pl.plot_dec_space(scaled_orig, dir1 + np.reshape(scaled_orig, (784)), dir2 + np.reshape(scaled_orig, (784)),
                          model,
                          show_legend=False, show_advs=show_advs)
        plt.title('contrast = %.2f' % (1 - 2 * epsilon[i]), fontdict={'fontsize': 17})

        plt.subplot(2, 1, 2)
        plt.imshow(scaled_orig, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])

        colors = ['tab:orange', 'tab:green', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:blue', 'tab:cyan', 'tab:olive',
                  'tab:red', 'tab:purple']
        labels = []
        for c in range(10):
            labels.append(mpatches.Patch(color=colors[c], label=str(c)))
        lg = f.legend(handles=labels, title='predicted class', loc='upper right', bbox_to_anchor=(0.95, 0.85),
                      prop={'size': 14})
        lg.get_title().set_fontsize(15)
        plt.subplots_adjust(right=0.9, left=0.02)

        plt.savefig('../data/dec_movie%d/%03d.png' % (k, i))
        plt.close()


def get_mean_dist_dec(orig, adv1, adv2, model, get_abs=False):
    dec_radius = 4.5
    n_angles = 100
    n_scales = 200

    dir1 = (adv1 - orig.reshape(784)) / np.linalg.norm(adv1 - orig.reshape(784))
    dir2 = (adv2 - orig.reshape(784)) / np.linalg.norm(adv2 - orig.reshape(784))
    label = np.argmax(model(torch.tensor(np.reshape(orig, (1, 1, 28, 28)))).detach().cpu().numpy())

    scales = np.linspace(0, dec_radius, n_scales).reshape((-1, 1))
    angles = np.linspace(0, np.pi / 2, n_angles)
    x = np.cos(angles)
    y = np.sin(angles)

    dists = 0
    for i in range(len(x)):
        x_dir = scales * x[i]
        y_dir = scales * y[i]
        input_dirs = np.reshape(x_dir, (-1, 1)) * dir1 + np.reshape(y_dir, (-1, 1)) * dir2
        input = input_dirs + orig.reshape(784)
        pred = model(torch.tensor(input.astype('float32')).reshape(-1, 1, 28, 28)).detach().cpu().numpy()
        pred_classes = np.argmax(pred, axis=-1)
        if np.all(pred_classes == label):
            dists += dec_radius
        else:
            idx = np.min(np.argwhere(pred_classes != label))
            dists += np.linalg.norm(input_dirs[idx])
    mean_dist = dists / len(x)
    if not get_abs:
        mean_dist /= dec_radius
    return mean_dist
