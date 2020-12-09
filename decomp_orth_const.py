import sys
sys.path.insert(0, 'AnalysisBySynthesis')

import numpy as np
import torch


import foolbox
from abs_models import models as mz, utils as u
from run_batch import run_batch

# own modules
from utils import load_data
from attacks import CarliniWagner

np.random.seed(0)
torch.manual_seed(0)

model = mz.get_CNN()                      # Vanilla CNN
model.eval()
fmodel = foolbox.models.PyTorchModel(model,
                                     bounds=(0., 1.),
                                     device=u.dev())

n_images = 20
images, labels = load_data(n_images, bounds=(0., 1.))

# user initialization
attack_params = {
        'binary_search_steps': 12,
        'initial_const': 1e-2,
        'steps': 500,
        'confidence': 1,
        'abort_early': True
    }
params = {
    'n_adv_dims': 2,
    'max_runs': 3,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'plot_loss': True
}

orth_consts = [5]
pert_lengths = []
advs = []

for orth_const in orth_consts:
    new_advs, _, _, new_pert_lengths = run_batch(fmodel,images,labels,attack_params,orth_const,**params)
    advs.append(new_advs)
    pert_lengths.append(new_pert_lengths)
data = {
    'advs':advs,
    'pert_lengths':pert_lengths
}
np.save('./data/orth_consts.npy', data)