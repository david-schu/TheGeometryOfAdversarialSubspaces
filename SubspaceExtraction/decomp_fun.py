import sys
sys.path.insert(0, './..')
sys.path.insert(0, '.')
import os

import numpy as np
import torch

# own modules
from utils import dev, load_stable_data, load_model
from SubspaceExtraction.attacks import L2OrthAttack
from SubspaceExtraction.run_attack import run_attack

if __name__ == "__main__":
    is_natural = int(sys.argv[1])
    batch_n = int(sys.argv[2])
    ## user initialization

    # set number of images per class for attack
    n_images = 10
    dset = 'CIFAR'

    # set attack parameters
    attack_params = {
            'binary_search_steps': 10,
            'initial_const': 1e-3,
            'steps': 200,
            'abort_early': True
        }

    # set hyperparameters
    params = {
        'n_adv_dims': 30,
        'early_stop': 3,
        'input_attack': L2OrthAttack,
        'random_start': False
    }

    # set seeds
    torch.manual_seed(0)

    # load a model
    if dset == 'MNIST':
        model_seed = 0  # only required for MNIST
        if is_natural:
            model_path = './../models/natural_' + str(model_seed) + '.pt'
            save_path = '../data/minst_natural_' + str(batch_n) + '.npy'
        else:
            model_path = './../models/robust_' + str(model_seed) + '.pt'
            save_path = '../data/mnist_robust_' + str(batch_n) + '.npy'
    elif dset == 'CIFAR':
        if is_natural:
            model_path = './../models/cifar_models/nat_diff_new.pt'
            save_path = '../data/cifar_wrn/cifar_natural_' + str(batch_n) + '.npy'
        else:
            model_path = './../models/cifar_models/rob_diff_new.pt'
            save_path = '../data/cifar_wrn/cifar_robust_' + str(batch_n) + '.npy'
    else:
        raise ValueError('No valid dataset specification')

    model = load_model(resume_path=model_path, dataset=dset)

    # load batched data
    data = load_stable_data(dset)
    all_images, all_labels = data['images'], data['labels']
    images = all_images[all_labels == 0][:n_images]
    labels = all_labels[all_labels == 0][:n_images]
    for l in np.arange(1, 10):
        images = torch.cat((images, all_images[all_labels == l][:n_images]), 0)
        labels = torch.cat((labels, all_labels[all_labels == l][:n_images]), 0)
    del all_images, all_labels

    image = images[batch_n]
    label = labels[batch_n]

    image = image.to(dev()).unsqueeze(0)
    label = label.to(dev()).unsqueeze(0)

    if os.path.exists(save_path):
        pre_data = np.load(save_path, allow_pickle=True).item()
    else:
        pre_data = None


    # run decomposition over batches
    advs, dirs, adv_class, pert_lengths = run_attack(model, image, label, attack_params, save_path, pre_data, **params)

    data = {
        'advs': advs.unsqueeze(0).cpu().detach().numpy(),
        'dirs': dirs.unsqueeze(0).cpu().detach().numpy(),
        'adv_class': adv_class.unsqueeze(0).cpu().detach().numpy(),
        'pert_lengths': pert_lengths.unsqueeze(0).cpu().detach().numpy(),
        'images': image.detach().cpu().numpy(),
        'labels': label.detach().cpu().numpy(),
    }

    np.save(save_path, data)