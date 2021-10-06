import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')

import numpy as np
import dill
import torch
from robustness.datasets import CIFAR

# own modules
from utils import dev
from attacks import CarliniWagner
from run_attack import run_attack
from models import model as md


## user initialization

# set number of images for attack and batchsize (shouldn't be larger than 20)
n_images = 1
pre_data = None
# d_set = 'MNIST'

# set attack parameters
attack_params = {
        'binary_search_steps': 10,
        'initial_const': 1e-1,
        'steps': 20,
        'abort_early': True
    }

# set hyperparameters
params = {
    'n_adv_dims': 30,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'random_start': False,
}

# set seeds
# np.random.seed(0)
torch.manual_seed(0)

# load a model
# model = md.madry_diff()
# # model.load_state_dict(torch.load('./../models/adv_trained_l2.pt', map_location=torch.device(dev())))      # madry robust model
# model.load_state_dict(torch.load('./../models/natural_0.pt', map_location=torch.device(dev())))      # natural cnn - same architecture as madry robust model but nmot adversarially trained


ds = CIFAR('../data/cifar-10-batches-py')
classifier_model = ds.get_model('resnet50', False)
model = md.cifar_pretrained(classifier_model, ds)

resume_path = '../models/cifar_l2_0_5.pt'
checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))

state_dict_path = 'model'
if not ('model' in checkpoint):
    state_dict_path = 'state_dict'
sd = checkpoint[state_dict_path]
sd = {k[len('module.'):]:v for k,v in sd.items()}
model.load_state_dict(sd)
model.to(dev())
model.double()
model.eval()

# load batched data
data = np.load('../data/CIFAR/stable_data.npy', allow_pickle=True).item()
all_images, all_labels = data['images'], data['labels']
images = all_images[all_labels==0][:n_images]
labels = all_labels[all_labels==0][:n_images]
for l in np.arange(1,10):
   images = torch.cat((images,all_images[all_labels==l][:n_images]),0)
   labels = torch.cat((labels, all_labels[all_labels == l][:n_images]), 0)

images = images.to(dev())
labels = labels.to(dev())

del all_images, all_labels
# params['init'] = images[450].detach().cpu().numpy().flatten()

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