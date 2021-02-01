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
data = np.load('../data/cnn5.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']

### laod models
model_natural = model.madry()
model_madry = model.madry()
model_natural.load_state_dict(torch.load('./../models/natural.pt', map_location=torch.device(dev())))
model_madry.load_state_dict(torch.load('./../models/adv_trained.pt', map_location=torch.device(dev())))

#### calculate index of mean adversarial
mean_pert_length = np.mean(pert_lengths, axis=0)
dist_to_mean = np.sum(np.abs(pert_lengths - mean_pert_length), axis=-1)
min_idx = np.argmin(dist_to_mean)

pl.plot_dec_space(images[min_idx], advs[min_idx,0], advs[min_idx,1], model_natural)
plt.show()
md = dec_space.get_mean_dist_dec(images[min_idx], advs[min_idx,0], advs[min_idx,1], model_natural)
# _, unique_idx = np.unique(labels, return_index=True)
# ##### plot cw surfaces for madry and natural cnn dirs and models
# for i in unique_idx:
#     pl.save_dec_movie(images[i], advs[i,0], advs[i,1], model_natural, n=200, k=i)

pl.plot_cw_surface(images[min_idx], advs[min_idx,0], advs[min_idx,1], model_natural)
pl.plot_dec_space(images[min_idx], advs[min_idx,0], advs[min_idx,1], model_natural)
# pl.plot_cw_surface(images[min_idx], advs[min_idx, 0], advs[min_idx, 1], model_madry)
pl.plot_dec_space(images[min_idx], advs[min_idx,0], advs[min_idx,1], model_madry)


#### plot madry and natural adversarial examples in one figure
# cnn = np.load('../data/cnn500.npy', allow_pickle=True).item()
# madry = np.load('../data/madry500.npy', allow_pickle=True).item()
# for i in range(5):
#     plt.subplot(2,5,1+i)
#     plt.title('Adv. class ' + str(cnn['adv_class'][i,0]))
#     plt.imshow(np.reshape(cnn['advs'][i,0], [28,28]), cmap='gray', vmin=0, vmax=1)
#     plt.xticks([])
#     plt.yticks([])
#     if i == 0:
#         plt.ylabel("Natural CNN")
#
#     plt.subplot(2,5,6+i)
#     plt.title('Adv. class ' + str(madry['adv_class'][i,0]))
#     plt.imshow(np.reshape(madry['advs'][i,0], [28,28]), cmap='gray', vmin=0, vmax=1)
#     plt.xticks([])
#     plt.yticks([])
#     if i == 0:
#         plt.ylabel("Madry CNN")
# plt.suptitle('Adversarials of non-robust and robust models')
# plt.show()

#### plot pert_length trajectories for different runs of one image
# pert_lengths[pert_lengths==0] = np.nan
# for p in pert_lengths:
#     plt.plot(range(1, pert_lengths.shape[1]+1), p)
# plt.xlabel('n')
# plt.ylabel('adversarial vector length ($l2-norm$)')
# plt.xticks(range(1, pert_lengths.shape[1]+1))
# plt.title('Perturbation length of fist 10 adversarial directions')
# plt.ylim(0.5)
# plt.show()


print('DONE!')
