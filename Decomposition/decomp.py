import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')

import numpy as np
import torch
import foolbox
from abs_models import models as mz

# own modules
from utils import load_data, dev
from attacks import CarliniWagner
from run_batch import run_batch
from models import model

## user initialization

# set number of images for attack and batchsize (shouldn't be larger than 20)
n_images = 500
batchsize = 10
load_pre_data =True
data_path = '/home/bethge/dschultheiss/AdversarialDecomposition/data/cnn.npy'
pre_data = None

# set attack parameters
attack_params = {
        'binary_search_steps': 9,
        'initial_const': 1e-2,
        'steps': 5000,
        'confidence': 1,
        'abort_early': True
    }

# set hyperparameters
params = {
    'n_adv_dims': 5,
    'max_runs': 50,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'random_start': True,
    'pre_data': pre_data
}

# set seeds
np.random.seed(0)
torch.manual_seed(0)

# load a model
model = model.madry()
model.load_state_dict(torch.load('./../models/natural.pt', map_location=torch.device(dev())))      # madry robust model
# model.load_state_dict(torch.load('./../models/adv_trained_l2.pt', map_location=torch.device(dev())))      # natural cnn - same architecture as madry robust model but nmot adversarially trained
# model = mz.get_VAE(n_iter=50)   # ABS model

# laod data
if load_pre_data:
    data_load = np.load(data_path, allow_pickle=True).item()
    pre_advs = torch.split(torch.tensor(data_load['advs'], device=dev()), batchsize, dim=0)
    pre_dirs = torch.split(torch.tensor(data_load['dirs'], device=dev()), batchsize, dim=0)
    pre_class = torch.split(torch.tensor(data_load['adv_class'], device=dev()), batchsize, dim=0)
    pre_pert_lengths = torch.split(torch.tensor(data_load['pert_lengths'], device=dev()), batchsize, dim=0)
    pre_adv_found = torch.split(torch.tensor(~np.isnan(data_load['pert_lengths']), device=dev()), batchsize, dim=0)
    images = torch.tensor(data_load['images'], device=dev())
    labels = torch.tensor(data_load['labels'], device=dev())

else:
    images, labels = load_data(n_images, bounds=(0., 1.))


model.eval()
fmodel = foolbox.models.PyTorchModel(model,   # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.), #num_classes=10,
                                 device=dev())

# load batched data
images, labels = load_data(n_images, bounds=(0., 1.))
batched_images = torch.split(images, batchsize, dim=0)
batched_labels = torch.split(labels, batchsize, dim=0)

# initialize data arrays
advs = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims'], batched_images[0].shape[-1]**2))
dirs = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims'], batched_images[0].shape[-1]**2))
pert_lengths = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims']))
adv_class = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims']))

# run decomposition over batches
for i in range(len(batched_images)):
    print('Batch %d of %d: %.0d%% done ...' % (i+1, len(batched_images), i*100/len(batched_images)))
    if load_pre_data:
        pre_data = {
            'advs': pre_advs[i],
            'adv_dirs': pre_dirs[i],
            'adv_class': pre_class[i],
            'pert_lengths': pre_pert_lengths[i],
            'adv_found': pre_adv_found[i]
        }
    new_advs, new_dirs, new_classes, new_pert_lengths, new_adv_found = run_batch(fmodel, batched_images[i],
                                                                                 batched_labels[i], attack_params,
                                                                                 **params, pre_data=pre_data)

    advs = torch.cat([advs, new_advs], 0)
    dirs = torch.cat([dirs, new_dirs], 0)
    adv_class = torch.cat([adv_class, new_classes], 0)
    pert_lengths = torch.cat([pert_lengths, new_pert_lengths], 0)
    adv_found = torch.cat([adv_found, new_adv_found], 0)

data = {
    'advs': advs.cpu().detach().numpy(),
    'dirs': dirs.cpu().detach().numpy(),
    'adv_class': adv_class.cpu().detach().numpy(),
    'pert_lengths': pert_lengths.cpu().detach().numpy(),
    'adv_found': adv_found.cpu().detach().numpy(),
    'images': images.cpu().detach().numpy(),
    'labels': labels.cpu().detach().numpy()
}
np.save('/home/bethge/dschultheiss/AdversarialDecomposition/data/cnn.npy', data)