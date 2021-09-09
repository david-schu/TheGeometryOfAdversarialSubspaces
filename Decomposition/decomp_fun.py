import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')

import numpy as np
import torch
import dill
from robustness.datasets import CIFAR

# own modules
from utils import load_data, dev
from attacks import CarliniWagner
from run_attack import run_attack
from models import model as md

if __name__ == "__main__":
    model_id = int(sys.argv[1])
    is_natural = int(sys.argv[2])
    batch_n = int(sys.argv[3])
    ## user initialization

    # set number of images for attack and batchsize (shouldn't be larger than 20)
    n_images = 10
    batchsize = 5
    pre_data = None

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
        model_path = './../models/cifar_nat.pt'
    else:
        model_path = './../models/cifar_l2_0_5.pt'

    ds = CIFAR('../data/cifar-10-batches-py')
    classifier_model = ds.get_model('resnet50', False)
    model = md.cifar_pretrained(classifier_model, ds)

    checkpoint = torch.load(model_path, pickle_module=dill, map_location=torch.device(dev()))
    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'
    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(dev())
    model.double()
    model.eval()

    # load batched data
    data = np.load('../data/CIFAR/stable_data.npy', allow_pickle=True).item()
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