import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')

import numpy as np
import torch

from tqdm import tqdm

# own modules
from utils import load_data, dev
from attacks import CarliniWagner
from run_attack import run_attack
from models import model as md

## user initialization

# set number of images for attack and batchsize (shouldn't be larger than 20)
n_images = 1
pre_data = None
d_set = 'MNIST'

# set attack parameters
attack_params = {
        'binary_search_steps': 5,
        'initial_const': 1e-2,
        'steps': 100,
        'abort_early': True
    }

# set hyperparameters
params = {
    'n_adv_dims': 30,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'random_start': False
}

# set seeds
np.random.seed(0)
torch.manual_seed(0)

# load a model
model = md.madry()
# model.load_state_dict(torch.load('./../models/adv_trained_l2.pt', map_location=torch.device(dev())))      # madry robust model
model.load_state_dict(torch.load('./../models/natural.pt', map_location=torch.device(dev())))      # natural cnn - same architecture as madry robust model but nmot adversarially trained

model.eval()

# load batched data
images, labels = load_data(n_images, bounds=(0., 1.), d_set=d_set, random=False)

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
np.save('/home/bethge/dschultheiss/AdversarialDecomposition/data/cnn_single_trust_reg.npy', data)