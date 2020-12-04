import sys
sys.path.insert(0, './../')
sys.path.insert(0, '/home/bethge/dschultheiss/AnalysisBySynthesis')

import torch

import numpy as np

import foolbox
from abs_models import models as mz
from abs_models import utils as u

# own modules
from utils import classification, dirs_to_attack_format, load_data
from attacks import OrthogonalAttack, CarliniWagner


# model = mz.get_VAE(n_iter=10)              # ABS, do n_iter=50 for original model
# model = mz.get_VAE(binary=True)           # ABS with scaling and binaryzation
# model = mz.get_binary_CNN()               # Binary CNN
model = mz.get_CNN()                      # Vanilla CNN
# model = mz.get_NearestNeighbor()          # Nearest Neighbor, "nearest L2 dist to each class"=logits
# model = mz.get_madry()                    # Robust network from Madry et al. in tf
# model = create()
model.eval()


fmodel = foolbox.models.PyTorchModel(model,   # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.), #num_classes=10,
                                     device=u.dev())
n_images = 20
images, labels = load_data(n_images, bounds=(0., 1.))

# user initialization
batchsize = 20
n_adv_dims = 10
max_runs = 1000
show_plots = False
early_stop = 3
norm_order = 2
steps = 10000
input_attack = CarliniWagner
epsilons = [0.3]

# variable initializations
attack_params = {
        'binary_search_steps':12,
        'initial_const':1e-2,
        'steps':steps,
        'confidence':1,
        'abort_early':True
    }

n_pixel = images.shape[-1]**2
x_orig = u.t2n(images).reshape([n_images,n_pixel])
orth_consts = [50]

for orth_const in orth_consts:
    count = 0
    min_dim = 0
    adv_dirs = []
    pert_lengths = []
    advs = []
    dirs = torch.tensor([])
    adv_dirs =[]
    adv_class = []

    for run in range(max_runs):
        print('Run %d - Adversarial Dimension %d...' % (run+1, min_dim + 1))

        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
                                  orth_const=orth_const,
                                  plot_loss=False)
        adv, _, success = attack(fmodel, images, labels, epsilons=epsilons)

        # check if adversarials were found and stop early if not
        if success.sum() == 0:
            print('--No attack within bounds found--')
            count +=1
            if early_stop == count:
                print('No more adversarials found ----> early stop!')
                break
            continue

        count = 0

        classes = classification(adv[0],model)
        min_dim = n_adv_dims
        for i, a in enumerate(adv[0]):
            a_ = u.t2n(a.flatten())
            pert_length = np.linalg.norm(a_ - x_orig[i], ord=2)
            if run == 0:
                min_dim = 1
                advs.append(np.array([a_]))
                pert_lengths.append(np.array([pert_length]))
                adv_dirs.append(np.array([(a_ - x_orig[i])/pert_length]))
                adv_class.append(np.array([classes[i]]))
            else:
                dim = np.sum(pert_lengths[i] < pert_length)
                min_dim = np.minimum(min_dim,dim)+1
                advs[i] = np.vstack([advs[i][:dim], a_])
                adv_dirs[i] = np.vstack([adv_dirs[i][:dim], (a_ - x_orig[i])/pert_length])
                adv_class[i] = np.append(adv_class[i][:dim], classes[i])
                pert_lengths[i] = np.append(pert_lengths[i][:dim], pert_length)
        dirs = dirs_to_attack_format(adv_dirs)
        if min_dim == n_adv_dims:
            break
data = {
    "advs": advs,
    "adv_dirs": adv_dirs,
    "adv_class": adv_class,
    "pert_lengths": pert_lengths,
    "images": images,
    "labels": labels
}

np.save('/home/bethge/dschultheiss/data/temp.npy', data)