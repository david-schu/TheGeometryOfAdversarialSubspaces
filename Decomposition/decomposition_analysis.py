import matplotlib.pyplot as plt
import numpy as np
import plots as pl
import dec_space
from models import model
import torch
from utils import dev
from torchvision import datasets, transforms
from models import eval
from abs_models import models as mz
from matplotlib.ticker import FormatStrFormatter
import time

### load necessary data
data = np.load('../data/cnn.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']
# dims = data['dims']




data = np.load('../data/robust.npy', allow_pickle=True).item()
advs_madry = data['advs']
pert_lengths_madry = data['pert_lengths']
classes_madry = data['adv_class']
dirs_madry = data['dirs']
images_madry = data['images']
labels_madry = data['labels']

### laod models
model_natural = model.madry()
model_madry = model.madry()
model_natural.load_state_dict(torch.load('./../models/natural.pt', map_location=torch.device(dev())))
model_madry.load_state_dict(torch.load('./../models/adv_trained_l2.pt', map_location=torch.device(dev())))

# eval_batch_size=200
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False,
#                    transform=transforms.ToTensor()),
#     batch_size=eval_batch_size)
# eval.evalAdvAttack(model_madry,test_loader,1.5)
# eval.evalAdvAttack(model_natural,test_loader,1.5)


# dists_natural_natural = np.load('../data/dists_natural_natural.npy', allow_pickle=True)
# dists_natural_madry = np.load('../data/dists_natural_madry.npy', allow_pickle=True)
# dists_madry_natural = np.load('../data/dists_madry_natural.npy', allow_pickle=True)
# dists_madry_madry = np.load('../data/dists_madry_madry.npy', allow_pickle=True)
#
#
# classes_madry[pert_lengths_madry==0]=np.nan
# pl.plot_var_hist(classes_madry[:,:5],labels_madry,'Madry CNN - 5 Directions')

#### get distance metrics for different models and advs
# dists_natural_natural = np.zeros(len(images))
# dists_natural_madry = np.zeros(len(images))
# dists_madry_natural = np.zeros(len(images))
# dists_madry_madry = np.zeros(len(images))
# for i in range(len(images)):
#     dists_natural_natural[i] = dec_space.get_mean_dist_dec(images[i], advs[i, 0], advs[i, 1], model_natural)
#     dists_natural_madry[i] = dec_space.get_mean_dist_dec(images[i], advs[i, 0], advs[i, 1], model_madry)
#     dists_madry_natural[i] = dec_space.get_mean_dist_dec(images_madry[i], advs_madry[i, 0], advs_madry[i, 1], model_natural)
#     dists_madry_madry[i] = dec_space.get_mean_dist_dec(images_madry[i], advs_madry[i, 0], advs_madry[i, 1], model_madry)
#
# np.save('../data/dists_natural_natural.npy', dists_natural_natural)
# np.save('../data/dists_natural_madry.npy', dists_natural_madry)
# np.save('../data/dists_madry_natural.npy', dists_madry_natural)
# np.save('../data/dists_madry_madry.npy', dists_madry_madry)


## plot grid of adversarial examples
fig, ax = plt.subplots(5, 6, squeeze=False, figsize=(9,8))
for j in range(5):
    orig = np.reshape(images[j], [28, 28])
    if j == 0:
        ax[j, 0].set_title('original', fontsize=18)
    ax[j, 0].imshow(orig, cmap='gray', vmin=0, vmax=1)
    ax[j, 0].set_xticks([])
    ax[j, 0].set_yticks([])
    ax[j, 0].set_xlabel(str(labels[j]), fontdict={'fontsize': 18})

    for i, a in enumerate(advs[j,:5]):
        if j==0:
            ax[j, i+1].set_title('Adv. ' + str(j + 1), fontsize=18)
        ax[j, i+1].set_xlabel('\u279E ' + str(int(classes[j,i])), fontdict={'fontsize': 18})
        ax[j, i+1].imshow(a.reshape([28, 28]), cmap='gray', vmin=0, vmax=1)
        ax[j, i+1].set_xticks([])
        ax[j, i+1].set_yticks([])
plt.subplots_adjust(hspace=0.3, left=0, right=1, bottom=0.05, top=0.95)
plt.show()

#### calculate index of mean adversarial
# mean_pert_length = np.mean(pert_lengths, axis=0)
# dist_to_mean = np.sum(np.abs(pert_lengths - mean_pert_length), axis=-1)
# min_idx = np.argmin(dist_to_mean)


# _, unique_idx = np.unique(labels, return_index=True)

##### plot cw surfaces for madry and natural cnn dirs and models
# for i in unique_idx:
#     plt.subplots(2,2)
#     plt.subplot(2,2,1)
#     pl.plot_dec_space(images[i], advs[i,0], advs[i,1], model_natural)
#     plt.subplot(2,2,2)
#     pl.plot_dec_space(images[i], advs[i, 0], advs[i, 1], model_madry)
#     plt.subplot(2,2,3)
#     pl.plot_dec_space(images_madry[i], advs_madry[i, 0], advs_madry[i, 1], model_natural)
#     plt.subplot(2,2,4)
#     pl.plot_dec_space(images_madry[i], advs_madry[i, 0], advs_madry[i, 1], model_madry)
#     plt.show()


#### plot madry and natural adversarial examples in one figure
# cnn = np.load('../data/cnn5.npy', allow_pickle=True).item()
# madry = np.load('../data/madry_l2.npy', allow_pickle=True).item()
# for x in [0,8,19,289]:
#     for i in range(5):
#         plt.subplot(2,5,1+i)
#         plt.title('Adv. class ' + str(cnn['adv_class'][x+i,0]))
#         plt.imshow(np.reshape(cnn['advs'][x+i,0], [28,28]), cmap='gray', vmin=0, vmax=1)
#         plt.xticks([])
#         plt.yticks([])
#         if i == 0:
#             plt.ylabel("Natural CNN")
#
#         plt.subplot(2,5,6+i)
#         plt.title('Adv. class ' + str(madry['adv_class'][x+i,0]))
#         plt.imshow(np.reshape(madry['advs'][x+i,0], [28,28]), cmap='gray', vmin=0, vmax=1)
#         plt.xticks([])
#         plt.yticks([])
#         if i == 0:
#             plt.ylabel("Madry CNN")
#     plt.suptitle('Adversarials of non-robust and robust models')
#     plt.show()

#### plot pert_length trajectories for different runs of one image
# pert_lengths[pert_lengths==0] = np.nan
# for p in pert_lengths[:,:8]:
#     plt.plot(range(1, 9), p)
# plt.xlabel('d')
# plt.ylabel('adversarial vector length ($l2-norm$)')
# plt.xticks(range(1, 9))
# # plt.title('Perturbation length of fist 10 adversarial directions')
# plt.ylim(0.5)
# plt.show()


print('DONE!')
