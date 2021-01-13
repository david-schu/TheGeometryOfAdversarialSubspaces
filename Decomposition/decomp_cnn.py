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

np.random.seed(0)
torch.manual_seed(0)

model = model.madry()
model.load_state_dict(torch.load('./../models/normal.pt', map_location=torch.device(dev())))
model.eval()
fmodel = foolbox.models.PyTorchModel(model,   # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.), #num_classes=10,
                                     device=dev())
n_images = 10
batchsize = 10
images, labels = load_data(n_images, bounds=(0., 1.))
batched_images = torch.split(images, batchsize, dim=0)
batched_labels = torch.split(labels, batchsize, dim=0)

# user initialization
attack_params = {
        'binary_search_steps':9,
        'initial_const':1e-2,
        'steps':5000,
        'confidence':1,
        'abort_early':True
    }
params = {
    'n_adv_dims': 5,
    'max_runs': 50,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'plot_loss': False,
    'random_start': True
}

advs = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims'], batched_images[0].shape[-1]**2))
dirs = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims'], batched_images[0].shape[-1]**2))
pert_lengths = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims']))
adv_class = torch.tensor([], device=dev()).reshape((0, params['n_adv_dims']))


for i in range(len(batched_images)):
    print('Batch %d of %d: %.0d%% done ...' % (i+1, len(batched_images), i*100/len(batched_images)))
    new_advs, new_dirs, new_classes, new_pert_lengths = run_batch(fmodel, batched_images[i], batched_labels[i], attack_params, **params)

    advs = torch.cat([advs, new_advs], 0)
    dirs = torch.cat([dirs, new_dirs], 0)
    adv_class = torch.cat([adv_class, new_classes], 0)
    pert_lengths = torch.cat([pert_lengths, new_pert_lengths], 0)

pert_lengths = pert_lengths.cpu().detach().numpy()
adv_class = adv_class.cpu().detach().numpy()
pert_lengths[pert_lengths == 0] = np.nan
adv_class[pert_lengths == 0] = np.nan

data = {
    'advs': advs.cpu().detach().numpy(),
    'dirs': dirs.cpu().detach().numpy(),
    'adv_class': adv_class,
    'pert_lengths': pert_lengths,
    'images': images.cpu().detach().numpy(),
    'labels': labels.cpu().detach().numpy()
}
np.save('/home/bethge/dschultheiss/AdversarialDecomposition/data/cnn.npy', data)