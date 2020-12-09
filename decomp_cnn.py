import sys
sys.path.insert(0, './../')
sys.path.insert(0, '/home/bethge/dschultheiss/AnalysisBySynthesis')


import numpy as np

import foolbox
from abs_models import models as mz, utils as u

# own modules
from utils import load_batched_data
from attacks import CarliniWagner
from run_batch import run_batch


model = mz.get_CNN()                      # Vanilla CNN
model.eval()
fmodel = foolbox.models.PyTorchModel(model,   # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.), #num_classes=10,
                                     device=u.dev())
n_images = 1000
batchsize = 20
images, labels = load_batched_data(n_images,batchsize, bounds=(0., 1.))

# user initialization
attack_params = {
        'binary_search_steps':12,
        'initial_const':1e-2,
        'steps':10000,
        'confidence':1,
        'abort_early':True
    }
params = {
    'n_adv_dims':5,
    'max_runs': 100,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'plot_loss': False
}

advs = np.array([], shape=(0, params['n_adv_dims'], images.shape[-1]**2))
dirs = np.array([], shape=(0, params['n_adv_dims'], images.shape[-1]**2))
pert_lengths = np.array([], shape=(0, params['n_adv_dims']))
adv_class = np.array([], shape=(0, params['n_adv_dims']))


for i in range(len(images)):

    new_advs, new_dirs, new_classes, new_pert_lengths = run_batch(fmodel, images[i], labels[i], attack_params, **params)

    advs = np.concatenate([advs, new_advs], axis=0)
    dirs = np.concatenate([dirs, new_dirs], axis=0)
    adv_class = np.concatenate([adv_class, new_classes], axis=0)
    pert_lengths = np.concatenate([pert_lengths, new_pert_lengths], axis=0)

data = {
    'advs': advs,
    'dirs': dirs,
    'adv_class': adv_class,
    'pert_lengths': pert_lengths
}
np.save('/home/bethge/dschultheiss/data/cnn.npy', data)