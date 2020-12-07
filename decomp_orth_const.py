import sys
sys.path.insert(0, './../')
sys.path.insert(0, '/home/bethge/dschultheiss/AnalysisBySynthesis')

import numpy as np

import foolbox
from abs_models import models as mz
from abs_models import utils as u
from run_batch import run_batch

# own modules
from utils import load_data
from attacks import CarliniWagner


model = mz.get_CNN()                      # Vanilla CNN
model.eval()
fmodel = foolbox.models.PyTorchModel(model,   # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.), #num_classes=10,
                                     device=u.dev())
n_images = 8
images, labels = load_data(n_images, bounds=(0., 1.))

# user initialization
attack_params = {
        'binary_search_steps': 12,
        'initial_const': 1e-2,
        'steps': 10000,
        'confidence': 1,
        'abort_early': True
    }
params = {
    'n_adv_dims': 2,
    'max_runs': 30,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'plot_loss': True
}

orth_consts = [5]
pert_lengths = []
advs = []

for orth_const in orth_consts:
    new_advs, _, _, new_pert_lengths = run_batch(fmodel,images,labels,attack_params,orth_const,**params)
    advs.append(np.array(new_advs))
    pert_lengths.append(np.array(new_pert_lengths))
data = {
    'advs':advs,
    'pert_lengths':pert_lengths
}
np.save('/home/bethge/dschultheiss/data/orth_consts.npy', data)