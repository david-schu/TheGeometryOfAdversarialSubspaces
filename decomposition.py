import sys
sys.path.insert(0, './../')
sys.path.insert(0, './../AnalysisBySynthesis')

import torch
import numpy as np
from random import randint

import foolbox

from abs_models import models as mz
from abs_models import utils as u

# own modules
from run_batch import run_batch
from attacks import CarliniWagner
import plots as p

# model = mz.get_VAE(n_iter=10)              # ABS, do n_iter=50 for original model
# model = mz.get_VAE(binary=True)           # ABS with scaling and binaryzation
model = mz.get_binary_CNN()               # Binary CNN
# model = mz.get_CNN()  # Vanilla CNN
# model = mz.get_NearestNeighbor()          # Nearest Neighbor, "nearest L2 dist to each class"=logits
# model = mz.get_madry()                    # Robust network from Madry et al. in tf
# model = create()


model.eval()
fmodel = foolbox.models.PyTorchModel(model,  # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.),  # num_classes=10,
                                     device=u.dev())

images, labels = foolbox.utils.samples(fmodel, dataset="mnist", batchsize=2)  # returns random batch as torch tensor
# rand = randint(0,19)
# images = images[rand].unsqueeze(0)
# labels = labels[rand].unsqueeze(0)
labels = labels.long()

# images = torch.load('data/images.pt')
# labels = torch.load('data/labels.pt')



# user initializations
attack_params = {
    'binary_search_steps': 12,
    'initial_const': 1e-2,
    'steps': 5000,
    'confidence': 1,
    'abort_early': True
}

params = {
    'n_adv_dims':2,
    'max_runs': 10,
    'early_stop': 3,
    'input_attack': CarliniWagner,
    'epsilons': [None],
    'plot_loss': True
}

show_plots = True
orth_consts = [50]

for orth_const in orth_consts:
    new_advs, new_dirs, new_classes, new_pert_lengths = run_batch(fmodel,images,labels,attack_params,orth_const,**params)


    # # visualization images
    # if show_plots:
    #     p.plot_advs(images[0][0].numpy(), advs[0], 5)
    #     p.show_orth(adv_dirs[0])
    #     p.plot_pert_lengths(adv_class[0], pert_lengths[0])
