import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')

import numpy as np
import torch

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
    n_images = 1
    pre_data = None
    d_set = 'MNIST'

    # set attack parameters
    attack_params = {
            'binary_search_steps': 10,
            'initial_const': 1e-1,
            'steps': 100,
            'abort_early': True
        }

    # set hyperparameters
    params = {
        'n_adv_dims': 5,
        'early_stop': 3,
        'input_attack': CarliniWagner,
        'random_start': False
    }

    # set seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # load a model
    if is_natural:
        model_path = './../models/natural_' + str(model_id) + '.pt'
    else:
        model_path = './../models/robust_' + str(model_id) + '.pt'
    model = md.madry_diff()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(dev())))      # natural cnn - same architecture as madry robust model but nmot adversarially trained
    model.eval()

    # load batched data
    images, labels = load_data(n_images,train=False, bounds=(0., 1.), d_set=d_set, random=False)
    images = images[batch_n*n_images:batch_n*n_images+n_images]
    labels = labels[batch_n*n_images:batch_n*n_images+n_images]
    print(len(images))

    # initialize data arrays
    advs = np.zeros((n_images, params['n_adv_dims'], images[0].shape[0], images[0].shape[-1]**2))
    dirs = np.zeros((n_images, params['n_adv_dims'], images[0].shape[0], images[0].shape[-1]**2))
    pert_lengths = np.zeros((n_images, params['n_adv_dims']))
    adv_class = np.zeros((n_images, params['n_adv_dims']))

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
        save_path = '/home/bethge/dschultheiss/AdversarialDecomposition/data/natural' + str(model_id) + '_'\
                    + str(batch_n) + '.npy'
    else:
        save_path = '/home/bethge/dschultheiss/AdversarialDecomposition/data/robust' + str(model_id)\
                    + '_' + str(batch_n) + '.npy'
    np.save(save_path, data)