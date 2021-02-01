import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')

import numpy as np
import torch
import foolbox

# own modules
from utils import load_data, dev
from attacks import CarliniWagner
from run_batch import run_batch
from models import model

## user initialization

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
    'n_adv_dims': 784,
    'max_runs': 2000,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'plot_loss': False,
    'random_start': True,
    'verbose': True,
    'save_dims': True
}

# set seeds
np.random.seed(369)
torch.manual_seed(369)

# load a model
model = model.madry()
model.load_state_dict(torch.load('./../models/natural.pt', map_location=torch.device(dev())))       # natural cnn - same architecture as madry robust model
# model.load_state_dict(torch.load('./../models/madry.pt', map_location=torch.device(dev())))      # madry cnn

model.eval()
fmodel = foolbox.models.PyTorchModel(model,   # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.), #num_classes=10,
                                 device=dev())

# load batched data
images, labels = load_data(100, bounds=(0., 1.))
_, unique_idx = np.unique(labels.cpu().detach().numpy(), return_index=True)
images = images[unique_idx]
labels = labels[unique_idx]

# run decomposition over batches
advs, dirs, adv_class, pert_lengths, adv_found, dims = run_batch(fmodel, images, labels, attack_params, **params)
pert_lengths = pert_lengths.cpu().detach().numpy()
advs = advs.cpu().detach().numpy()
dirs = dirs.cpu().detach().numpy()
adv_class = adv_class.cpu().detach().numpy()

# min_dim = np.min(np.sum(~(pert_lengths==0), axis=-1))

# save data
data = {
    'advs': advs,
    'dirs': dirs,
    'adv_class': adv_class,
    'pert_lengths': pert_lengths,
    'adv_found': adv_found.cpu().detach().numpy(),
    'dims': dims,
    'images': images.cpu().detach().numpy(),
    'labels': labels.cpu().detach().numpy()
}
np.save('/home/bethge/dschultheiss/AdversarialDecomposition/data/cnn_long.npy', data)