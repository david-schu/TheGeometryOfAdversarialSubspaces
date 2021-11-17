#!/usr/bin/env python
# coding: utf-8

import sys

#import subprocess
#
#def pip_install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#
#proc = subprocess.Popen('apt-get install -y divpng', shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=STDOUT, executable="/bin/bash")
#proc.wait()
#proc = subprocess.Popen('apt-get install -y divpng', shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=STDOUT, executable="/bin/bash")
#proc.wait()


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import proplot as pplt
import pandas as pd

import numpy as np

ROOT = '/home/bethge/dpaiton/Work/AdversarialDecomposition/'
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT+'/data')
import plots as pl
import experiment_parser as exp
from curve_utils import tab_name_to_hex, load_mnist, load_cifar



# # Parameters

# In[2]:

dataset_type = 1
num_images = 50


# In[3]:


plot_settings = {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 8,#12,
        "axes.formatter.use_mathtext": False,
}
pplt.rc.update(plot_settings)
mpl.rcParams.update(plot_settings)

figwidth = '13.968cm'
figwidth_inch = 5.50107
dpi = 600
model_types = ['natural', 'robust']
plot_colors = [tab_name_to_hex('tab:blue'), tab_name_to_hex('tab:red')]


# # Load data

# In[ ]:


if dataset_type == 0: # MNIST
    #model_natural, data_natural, model_robust, data_robust = load_mnist(code_directory='../../', seed=0)
    data_name = 'mnist'
else: # CIFAR
    #model_natural, data_natural, model_robust, data_robust = load_cifar(code_directory='../../')
    data_name = 'cifar'

def load_runs(dataset_type, num_images):
    run_outputs = []
    for run_type in range(2):
        run_outputs.append(exp.get_combined_experiment_outputs(dataset_type, run_type, num_images))
        if 'data' in run_outputs[-1].keys():
            del run_outputs[-1]['data']
    exp_num_images, num_advs, num_pixels, num_directions = run_outputs[0]['principal_directions'].shape
    assert exp_num_images == num_images

    paired_principal_curvatures = np.stack([
        run_outputs[0]['principal_curvatures'], # natural, dataset pair
        run_outputs[1]['principal_curvatures']], axis=0) # robust, dataset pair
    paired_principal_directions = np.stack([
        run_outputs[0]['principal_directions'], # natural, dataset pair
        run_outputs[1]['principal_directions']], axis=0) # robust, dataset pair
    paired_mean_curvatures = np.mean(paired_principal_curvatures, axis=-1)
    paired_origin_indices = np.stack([
        run_outputs[0]['origin_indices'], # natural, dataset pair
        run_outputs[1]['origin_indices']], axis=0) #robust, dataset pair

    run_outputs = []
    for run_type in range(2, 4):
        run_outputs.append(exp.get_combined_experiment_outputs(dataset_type, run_type, num_images))
        if 'data' in run_outputs[-1].keys():
            del run_outputs[-1]['data']

    adv_principal_curvatures = np.stack([
        run_outputs[0]['principal_curvatures'], # natural, adversarial pair
        run_outputs[1]['principal_curvatures']], axis=0) # robust, adversarial pair
    adv_principal_directions = np.stack([
        run_outputs[0]['principal_directions'], # natural, adversarial pair
        run_outputs[1]['principal_directions']], axis=0) # robust, adversarial pair
    adv_mean_curvatures = np.mean(adv_principal_curvatures, axis=-1)
    adv_origin_indices = np.stack([
        run_outputs[0]['origin_indices'], # natural, adversarial pair
        run_outputs[1]['origin_indices']], axis=0) #robust, adversarial pair

    run_outputs = []
    for run_type in range(4, 6):
        run_outputs.append(exp.get_combined_experiment_outputs(dataset_type, run_type, num_images))
        if 'data' in run_outputs[-1].keys():
            del run_outputs[-1]['data']
    rand_subspace_pcs = np.stack([
        run_outputs[0]['principal_curvatures'], # natural, random subspace
        run_outputs[1]['principal_curvatures']], axis=0) # robust, random subspace
    rand_subspace_pds = np.stack([
        run_outputs[0]['principal_directions'], # natural, random subspace
        run_outputs[1]['principal_directions']], axis=0) # robust, random subspace

    run_outputs = []
    for run_type in range(6, 8):
        run_outputs.append(exp.get_combined_experiment_outputs(dataset_type, run_type, num_images))
        if 'data' in run_outputs[-1].keys():
            del run_outputs[-1]['data']
    adv_subspace_pcs = np.stack([
        run_outputs[0]['principal_curvatures'], # natural, adversarial subspace
        run_outputs[1]['principal_curvatures']], axis=0) # robust, adversarial subspace
    adv_subspace_pds = np.stack([
        run_outputs[0]['principal_directions'], # natural, adversarial subspace
        run_outputs[1]['principal_directions']], axis=0) # robust, adversarial subspace
    output = (
        paired_principal_curvatures, paired_principal_directions, paired_mean_curvatures, paired_origin_indices,
        adv_principal_curvatures, adv_principal_directions, adv_mean_curvatures, adv_origin_indices,
        rand_subspace_pcs, rand_subspace_pds, adv_subspace_pcs, adv_subspace_pds)
    return output

(paired_principal_curvatures, paired_principal_directions, paired_mean_curvatures, paired_origin_indices,
    adv_principal_curvatures, adv_principal_directions, adv_mean_curvatures, adv_origin_indices,
    rand_subspace_pcs, rand_subspace_pds, adv_subspace_pcs, adv_subspace_pds) = load_runs(dataset_type, num_images)
num_models, exp_num_images, num_advs, num_pixels, num_directions = paired_principal_directions.shape


# In[ ]:


bar_width = 0.5
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(figwidth_inch, (12/7)*figwidth_inch))
fig.subplots_adjust(top=0.8)

for data_idx, mean_curvatures in enumerate([paired_mean_curvatures, adv_mean_curvatures]):
    for model_idx in range(2):
        boxprops = dict(color=plot_colors[model_idx], linewidth=1.5, alpha=0.7)
        whiskerprops = dict(color=plot_colors[model_idx], alpha=0.7)
        capprops = dict(color=plot_colors[model_idx], alpha=0.7)
        medianprops = dict(linestyle='--', linewidth=0.5, color=plot_colors[model_idx])
        meanpointprops = dict(marker='o', markeredgecolor='black',
                              markerfacecolor=plot_colors[model_idx])
        meanprops = dict(linestyle='-', linewidth=0.5, color=plot_colors[model_idx])
        data = mean_curvatures[model_idx, :, :].reshape(-1)
        axs[data_idx].boxplot(data, sym='', positions=[model_idx], whis=(10, 90), widths=bar_width, meanline=True, showmeans=True, boxprops=boxprops,
            whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops, meanprops=meanprops)
    axs[data_idx].set_xticks([0, 1], minor=False)
    axs[data_idx].set_xticks([], minor=True)
    axs[data_idx].set_xticklabels(model_types)
    if data_idx == 0:
        axs[data_idx].set_ylabel('mean curvature')
        axs[data_idx].set_title('paired image boundary')
    else:
        axs[data_idx].set_title('adversarial image boundary')

for model_idx in range(adv_mean_curvatures.shape[0]):
    boxprops = dict(color=plot_colors[model_idx], linewidth=1.5, alpha=0.7)
    whiskerprops = dict(color=plot_colors[model_idx], alpha=0.7)
    capprops = dict(color=plot_colors[model_idx], alpha=0.7)
    medianprops = dict(linestyle='--', linewidth=0.5, color=plot_colors[model_idx])
    meanpointprops = dict(marker='o', markeredgecolor='black',
                          markerfacecolor=plot_colors[model_idx])
    meanprops = dict(linestyle='-', linewidth=0.5, color=plot_colors[model_idx])
    for adv_idx in range(adv_mean_curvatures.shape[-1]):
        data = adv_mean_curvatures[model_idx, :, adv_idx].reshape(-1)
        axs[2].boxplot(data, sym='', positions=[adv_idx], whis=(10, 90), widths=bar_width, meanline=True, showmeans=True, boxprops=boxprops,
            whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops, meanprops=meanprops)
axs[2].set_title('adversarial image boundary')
axs[2].set_xlabel('dimension number')
axs[2].set_xticks([i for i in range(adv_mean_curvatures.shape[-1])], minor=False)
axs[2].set_xticks([], minor=True)
axs[2].set_xticklabels([str(i+1) for i in range(adv_mean_curvatures.shape[-1])])

def make_space_above(axes, topmargin=1):
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes
        obtained from: https://stackoverflow.com/a/55768955/
    """
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1 - s.top) * h + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

make_space_above(axs, topmargin=0.5)

fig.suptitle(f'curvature at the decision boundary\nfor {num_images} images and the first {num_advs} adversarial directions', y=1.0)
#plt.show()
fig.savefig(ROOT+f'/data/mean_curvature_boxplots.png', transparent=True, bbox_inches='tight', pad_inches=0.01)


# In[ ]:


bar_width = 0.5
fig, axs = pplt.subplots(nrows=1, ncols=2, sharey=True, figwidth=figwidth)
titles = ['test image boundary', 'adversarial image boundary']
for data_idx, mean_curvatures in enumerate([paired_mean_curvatures, adv_mean_curvatures]):
    data = pd.DataFrame(mean_curvatures.reshape(-1, np.prod(mean_curvatures.shape[1:])).transpose(1, 0),
                        columns=pd.Index(model_types, name=''))
    axs[data_idx].boxplot(data, fill=True, mean=True,
                          cycle=pplt.Cycle(plot_colors),
                          linewidth=0.5,
                          meanlinestyle='-', medianlinestyle='--',
                          marker='o', markersize=1.0
                         )
    axs[data_idx].format(
        xticklabels=['',''],#model_types,
        ylabel='mean curvature',
        title=titles[data_idx],
        xgrid=False
    )
    axs[data_idx].axhline(0.0, color='black', linestyle='dashed', linewidth=0.5)


axs.format(
    suptitle=f'curvature at the decision boundary',#\naveraged across {num_images} images and the first {num_advs} adversarial directions'
)

#pplt.show()
fig.savefig(ROOT+f'/data/mean_curvature_boxplots.png', transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=dpi)


# In[ ]:


num_models, num_images, num_advs, num_dims = adv_principal_curvatures.shape

fig, ax = pplt.subplots(nrows=1, ncols=1, figwidth=figwidth_inch/2, dpi=dpi, sharey=False, sharex=False)
for image_idx in range(num_images):
    for adv_idx in range(num_advs):
        ax.scatter(adv_principal_curvatures[0, image_idx, adv_idx, :],
                   s=0.01, c=plot_colors[0])
        ax.scatter(adv_principal_curvatures[1, image_idx, adv_idx, :],
                   s=0.01, c=plot_colors[1])

ix = ax.inset(
    bounds=[200, 0.50, 400, 1.5],
    transform='data', zoom=True,
    zoom_kw={'edgecolor': 'k', 'lw': 1, 'ls': '--'}
)
ix.format(
    xlim=(0, num_dims), ylim=(-0.02, 0.02), metacolor='red7',
    grid=False,
    linewidth=1.5, ticklabelweight='bold'
)
ix.plot([0, num_dims], [0, 0], lw=0.1, c='k')
ix.scatter(adv_principal_curvatures[0, ...].mean(axis=(0, 1)),
           s=0.005, alpha=1.0, c=plot_colors[0])
ix.scatter(adv_principal_curvatures[1, ...].mean(axis=(0, 1)),
           s=0.005, alpha=1.0, c=plot_colors[1])

ax.format(
    title=f'curvature profile, averaged across {num_images} images',
    xlim=(-5, num_dims+5),
    ylabel='curvature',
    xlabel='principal curvature direction',
    grid=False
)
for ax_loc in ['top', 'right']:
    ax.spines[ax_loc].set_color('none')
#pplt.show()

fig.savefig(ROOT+f'/data/curvature_profile.png', transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=dpi)


# In[ ]:


bad_images = []
for image_idx in range(num_images):
    if (np.any(np.isnan(rand_subspace_pcs[:, image_idx, ...]))
            or
            np.any(np.isnan(adv_subspace_pcs[:, image_idx, ...]))
            or
            np.any(np.isnan(rand_subspace_pds[:, image_idx, ...]))
            or
            np.any(np.isnan(adv_subspace_pds[:, image_idx, ...]))
           ):
        bad_images.append(image_idx)
rand_subspace_pcs = np.delete(rand_subspace_pcs, bad_images, axis=1)
adv_subspace_pcs = np.delete(adv_subspace_pcs, bad_images, axis=1)
rand_subspace_pds = np.delete(rand_subspace_pds, bad_images, axis=1)
adv_subspace_pds = np.delete(rand_subspace_pds, bad_images, axis=1)


# In[ ]:


fig, axs = pplt.subplots(nrows=1, ncols=2, figwidth=figwidth_inch, dpi=dpi)

axs[0].scatter(rand_subspace_pcs[0, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[0],
               bardata=np.std(rand_subspace_pcs[0, ...], axis=(0,1)), barc=plot_colors[0], barlw=0.5, capsize=0.0,)
axs[0].scatter(rand_subspace_pcs[1, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[1],
               bardata=np.std(rand_subspace_pcs[1, ...], axis=(0,1)), barc=plot_colors[1], barlw=0.5, capsize=0.0,)
axs[0].axhline(0.0, color='black', linestyle='dashed', linewidth=0.5)
axs[0].format(
    title=f'random subspaces'
)
for ax_loc in ['top', 'right']:
    axs[0].spines[ax_loc].set_color('none')

axs[1].scatter(adv_subspace_pcs[0, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[0],
               bardata=np.std(adv_subspace_pcs[0, ...], axis=(0,1)), barc=plot_colors[0], barlw=0.5, capsize=0.0,)
axs[1].scatter(adv_subspace_pcs[1, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[1],
               bardata=np.std(adv_subspace_pcs[1, ...], axis=(0,1)), barc=plot_colors[1], barlw=0.5, capsize=0.0,)
axs[1].axhline(0.0, color='black', linestyle='dashed', linewidth=0.5)
axs[1].format(
    title=f'adversarial subspaces'
)
for ax_loc in ['top', 'right']:
    axs[1].spines[ax_loc].set_color('none')

axs.format(
    ylabel='curvature',
    xlabel='principal curvature directions',
    grid=False
)

legend_handles = [mpatches.Patch(color=plot_colors[0], label='natural'),
                  mpatches.Patch(color=plot_colors[1], label='robust')]
axs[0].legend(handles=legend_handles, loc='upper right', ncols=1, frame=False)

#pplt.show()

fig.savefig(ROOT+f'/data/subspace_curvatures.png', transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=dpi)


# In[ ]:


array = [
    [6,  1,  2,],
    [3,  4,  5,]
]
hspace = [3]
wspace = [5, 5]

fig, axs = pplt.subplots(array, sharey=False, sharex=False,
                         dpi=dpi, figwidth=figwidth, hspace=hspace, wspace=wspace)

all_mean_curvatures_max = np.max([paired_mean_curvatures.max(), adv_mean_curvatures.max()])
all_mean_curvatures_min = np.min([paired_mean_curvatures.min(), adv_mean_curvatures.min()])
titles = ['test boundary', 'adversarial boundary']
ax = axs[:2]
for data_idx, mean_curvatures in enumerate([paired_mean_curvatures, adv_mean_curvatures]):
    data = pd.DataFrame(mean_curvatures.reshape(-1, np.prod(mean_curvatures.shape[1:])).transpose(1, 0),
                        columns=pd.Index(model_types, name=''))
    ax[data_idx].boxplot(data, mean=True,
                         cycle=pplt.Cycle(plot_colors),
                         fill=True,
                         linewidth=0.5,
                         meanlinestyle='-', medianlinestyle='--',
                         marker='o', markersize=1.0
                         )
    ax[data_idx].format(title=titles[data_idx])
    ax[data_idx].axhline(0.0, color='black', linestyle='dashed', linewidth=0.5)

ax.format(
    xticks=[],
    xticklabels=[],
    ylim=(np.round(all_mean_curvatures_min, 1), np.round(all_mean_curvatures_max, 1)),
    yticks=[-0.5, 0.0, 0.5],
    yticklabels=['-0.5', '0.0', '0.5'],
    xgrid=False,
)
ax[0].format(ylabel='mean curvature')

ax = axs[2]
num_models, num_images, num_advs, num_dims = adv_principal_curvatures.shape
max_pc = adv_principal_curvatures.max()
min_pc = adv_principal_curvatures.min()

for image_idx in range(num_images):
    for adv_idx in range(num_advs):
        ax.scatter(adv_principal_curvatures[0, image_idx, adv_idx, :],
                   s=0.01, c=plot_colors[0])
        ax.scatter(adv_principal_curvatures[1, image_idx, adv_idx, :],
                   s=0.01, c=plot_colors[1])
ix = ax.inset(
    bounds=[500, -100, 1500, 80],
    transform='data', zoom=True,
    zoom_kw={'edgecolor': 'k', 'lw': 1, 'ls': '--'}
)
ix.format(
    xlim=(0, num_dims), ylim=(-2, 2), metacolor='red7',
    xticks=[1, 1500, 3000],
    xticklabels=[],
    ytickloc='right',
    yticklabelloc='right',
    yformatter='%d',
    grid=False,
    linewidth=1.0, ticklabelweight='bold'
)
ix.plot([0, num_dims], [0, 0], lw=0.1, c='k')
ix.scatter(adv_principal_curvatures[0, ...].mean(axis=(0, 1)),
           s=0.005, alpha=1.0, c=plot_colors[0])
ix.scatter(adv_principal_curvatures[1, ...].mean(axis=(0, 1)),
           s=0.005, alpha=1.0, c=plot_colors[1])
ax.format(
    title=f'adversarial boundary',#,\nfull dimensionality',
    xlim=(-5, num_dims+5),
    xticks=(1, 1500, 3000),
    ylim=(min_pc, max_pc),
    ylabel='curvature',
    grid=False
)

max_sub_pc = np.max([np.max(rand_subspace_pcs), np.max(adv_subspace_pcs)])
min_sub_pc = np.min([np.min(rand_subspace_pcs), np.min(adv_subspace_pcs)])
ax = axs[3:5]
ax[0].scatter(rand_subspace_pcs[0, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[0],
               bardata=np.std(rand_subspace_pcs[0, ...], axis=(0,1)), barc=plot_colors[0], barlw=0.5, capsize=0.0,)
ax[0].scatter(rand_subspace_pcs[1, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[1],
               bardata=np.std(rand_subspace_pcs[1, ...], axis=(0,1)), barc=plot_colors[1], barlw=0.5, capsize=0.0,)
ax[0].axhline(0.0, color='black', linestyle='dashed', linewidth=0.5)
ax[0].format(title=f'random subspaces')#, titlepad=-5)

ax[1].scatter(adv_subspace_pcs[0, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[0],
               bardata=np.std(adv_subspace_pcs[0, ...], axis=(0,1)), barc=plot_colors[0], barlw=0.5, capsize=0.0,)
ax[1].scatter(adv_subspace_pcs[1, ...].mean(axis=(0,1)), s=2.0, c=plot_colors[1],
               bardata=np.std(adv_subspace_pcs[1, ...], axis=(0,1)), barc=plot_colors[1], barlw=0.5, capsize=0.0,)
ax[1].axhline(0.0, color='black', linestyle='dashed', linewidth=0.5)
ax[1].format(title=f'adversarial subspaces')#, titlepad=-5)

ax.format(
    xticks=[i for i in range(num_advs)],
    xticklabels=[f'{i+1:d}' for i in range(num_advs)],
    xtickminor=False,
    ylim=(-10, 10),#(min_sub_pc, max_sub_pc),
    grid=False
)

axs[3].format(xlabel='principal curvature directions')

for ax_loc in ['top', 'right']:
    for ax in axs:
        ax.spines[ax_loc].set_color('none')

legend_handles = [mpatches.Patch(color=plot_colors[0], label=model_types[0]),
                  mpatches.Patch(color=plot_colors[1], label=model_types[1])]
legend = axs[-1].legend(legend_handles,
               title='training type',
               columnspacing=-1., markerfirst=True,
               loc='fill', frame=False, ncols=1, pad=-8)
#for handle in legend.legendHandles:
#    handle.set_width(4.0)

axs[:-1].format(abc='A.', abcloc='ul')

#pplt.show()

fig.savefig(ROOT+f'/data/{data_name}_curvature_analysis.pdf', transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=dpi)
fig.savefig(ROOT+f'/data/{data_name}_curvature_analysis.png', transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=dpi)
