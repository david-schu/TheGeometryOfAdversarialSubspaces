import sys
sys.path.insert(0, './../')

import torch
import numpy as np
from random import randint

import foolbox

from abs_models import models as mz

# own modules
from utils import classification, dirs_to_attack_format
from attacks import OrthogonalAttack, CarliniWagner
import plots as p

# model = mz.get_VAE(n_iter=10)              # ABS, do n_iter=50 for original model
# model = mz.get_VAE(binary=True)           # ABS with scaling and binaryzation
model = mz.get_binary_CNN()               # Binary CNN
# model = mz.get_CNN()  # Vanilla CNN
# model = mz.get_NearestNeighbor()          # Nearest Neighbor, "nearest L2 dist to each class"=logits
# model = mz.get_madry()                    # Robust network from Madry et al. in tf
# model = create()


model.eval()
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
fmodel = foolbox.models.PyTorchModel(model,  # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.),  # num_classes=10,
                                     device=dev)

# images, labels = foolbox.utils.samples(fmodel, dataset="mnist", batchsize=2)  # returns random batch as torch tensor
# # rand = randint(0,19)
# # images = images[rand].unsqueeze(0)
# # labels = labels[rand].unsqueeze(0)
# labels = labels.long()

images = torch.load('data/images.pt')
labels = torch.load('data/labels.pt')

# user initialization
n_adv_dims = 3
max_runs = 10
show_plots = True
early_stop = 3
norm_order = 2
steps = 500
input_attack = CarliniWagner
epsilons = [None]

# variable initializations
attack_params = {
    'binary_search_steps': 9,
    'initial_const': 1e-2,
    'steps': steps,
    'confidence': 1,
    'abort_early': True
}

n_images = len(images)
n_pixel = images.shape[-1] ** 2
x_orig = images.numpy().reshape([n_images, n_pixel])
orth_consts = [50]

for orth_const in orth_consts:

    # initialize variables
    count = 0
    min_dim = 0
    adv_dirs = []
    pert_lengths = []
    advs = []
    dirs = torch.tensor([])
    adv_dirs = []
    adv_class = []

    for run in range(max_runs):
        print('Run %d - Adversarial Dimension %d...' % (run + 1, min_dim + 1))

        attack = OrthogonalAttack(input_attack=input_attack, params=attack_params, adv_dirs=dirs, orth_const=orth_const)
        adv, _, success = attack(fmodel, images, labels, epsilons=epsilons)

        # check if adversarials were found and stop early if not
        if success.sum() == 0:
            print('--No attack within bounds found--')
            count += 1
            if early_stop == count:
                print('No more adversarials found ----> early stop!')
                break
            continue

        count = 0

        classes = classification(adv[0], model)
        min_dim = n_adv_dims

        # save found adversarials and check if they are smaller than previously found adversarials
        for i, a in enumerate(adv[0]):
            a_ = a.flatten().numpy()
            pert_length = np.linalg.norm(a_ - x_orig[i], ord=2)
            if run == 0:
                min_dim = 1
                advs.append(np.array([a_]))
                pert_lengths.append(np.array([pert_length]))
                adv_dirs.append(np.array([(a_ - x_orig[i]) / pert_length]))
                adv_class.append(np.array([classes[i]]))
            else:
                dim = np.sum(pert_lengths[i] < pert_length)
                min_dim = np.minimum(min_dim, dim) + 1
                advs[i] = np.vstack([advs[i][:dim], a_])
                adv_dirs[i] = np.vstack([adv_dirs[i][:dim], (a_ - x_orig[i]) / pert_length])
                adv_class[i] = np.append(adv_class[i][:dim], classes[i])
                pert_lengths[i] = np.append(pert_lengths[i][:dim], pert_length)

        # convert adversarial directions to attack format
        dirs = dirs_to_attack_format(adv_dirs)

        # break if n-dim is reached
        if min_dim == n_adv_dims:
            break

    # visualization images
    if show_plots:
        p.plot_advs(images[0][0].numpy(), advs[0], 5)
        p.show_orth(adv_dirs[0])
        p.plot_pert_lengths(adv_class[0], pert_lengths[0])
