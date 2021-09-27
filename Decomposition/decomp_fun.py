import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')

import numpy as np
import torch
from robustness1.datasets import CIFAR
from robustness1 import model_utils

# own modules
from utils import  dev
from attacks import CarliniWagner
from run_attack import run_attack
from models import model as md

if __name__ == "__main__":
    batchsize = int(sys.argv[1])
    is_natural = int(sys.argv[2])
    batch_n = int(sys.argv[3])
    ## user initialization

    # set number of images per class for attack
    n_images = 10

    # set attack parameters
    attack_params = {
            'binary_search_steps': 10,
            'initial_const': 1e-3,
            'steps': 200,
            'abort_early': True
        }

    # set hyperparameters
    params = {
        'n_adv_dims': 50,
        'early_stop': 3,
        'input_attack': CarliniWagner,
        'random_start': False
    }

    # set seeds
    # np.random.seed(0)
    torch.manual_seed(0)

    # load a model
    # if is_natural:
    #     model_path = './../models/natural_' + str(model_id) + '.pt'
    # else:
    #     model_path = './../models/robust_' + str(model_id) + '.pt'
    # model = md.madry_diff()
    # model.load_state_dict(torch.load(model_path, map_location=torch.device(dev())))      # natural cnn - same architecture as madry robust model but nmot adversarially trained

    if is_natural:
        model_path = './../models/cifar_models/nat_diff.pt'
    else:
        model_path = './../models/cifar_models/rob_diff.pt'

    ds = CIFAR('../data/cifar-10-batches-py')
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=model_path)
    model.to(dev())
    model.double()
    model.eval()

    # load batched data
    data = np.load('../data/CIFAR/stable_data_diff.npy', allow_pickle=True).item()
    all_images, all_labels = data['images'], data['labels']
    images = all_images[all_labels == 0][:n_images]
    labels = all_labels[all_labels == 0][:n_images]
    for l in np.arange(1, 10):
        images = torch.cat((images, all_images[all_labels == l][:n_images]), 0)
        labels = torch.cat((labels, all_labels[all_labels == l][:n_images]), 0)
    del all_images, all_labels

    images = images[(batch_n*batchsize):(batch_n*batchsize+batchsize)]
    labels = labels[(batch_n*batchsize):(batch_n*batchsize+batchsize)]

    images = images.to(dev())
    labels = labels.to(dev())

    # initialize data arrays
    advs = np.zeros((batchsize, params['n_adv_dims'], images[0].shape[0] * images[0].shape[-1]**2))
    dirs = np.zeros((batchsize, params['n_adv_dims'], images[0].shape[0] * images[0].shape[-1]**2))
    pert_lengths = np.zeros((batchsize, params['n_adv_dims']))
    adv_class = np.zeros((batchsize, params['n_adv_dims']))

    # run decomposition over batches
    for i in range(len(images)):
        new_advs, new_dirs, new_classes, new_pert_lengths = run_attack(model, images[i].unsqueeze(0), labels[i].unsqueeze(0),
                                                                       attack_params, **params)
        advs[i] = new_advs.cpu().detach().numpy()
        dirs[i] = new_dirs.cpu().detach().numpy()
        adv_class[i] = new_classes.cpu().detach().numpy()
        pert_lengths[i] = new_pert_lengths.cpu().detach().numpy()

        data = {
            'advs': advs,
            'dirs': dirs,
            'adv_class': adv_class,
            'pert_lengths': pert_lengths,
            'images': images.detach().cpu().numpy(),
            'labels': labels.detach().cpu().numpy(),
        }

        if is_natural:
            save_path = '../data/cifar_natural_' + str(batch_n) + '.npy'
        else:
            save_path = '../data/cifar_robust_' + str(batch_n) + '.npy'
        np.save(save_path, data)