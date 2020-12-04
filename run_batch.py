import torch
import numpy as np
from attacks import OrthogonalAttack,CarliniWagner
from utils import classification, dirs_to_attack_format
from abs_models import utils as u

def run_batch(fmodel,
              images,
              labels,
              attack_params,
              orth_const=50,
              input_attack=CarliniWagner,
              n_adv_dims=3,
              max_runs=100,
              early_stop=3,
              epsilons=[None],
              plot_loss=False
    ):

    # initialize variables
    n_pixel = images.shape[-1] ** 2
    n_images = images.shape[0]
    x_orig = u.t2n(images).reshape([n_images, n_pixel])

    count = 0
    min_dim = 0
    pert_lengths = []
    advs = []
    dirs = torch.tensor([])
    adv_dirs = []
    adv_class = []

    for run in range(max_runs):
        print('Run %d - Adversarial Dimension %d...' % (run + 1, min_dim + 1))

        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
                                  orth_const=orth_const,
                                  plot_loss=False)
        adv, _, success = attack(fmodel, images, labels, epsilons=epsilons)

        # check if adversarials were found and stop early if not
        if success.sum() == 0:
            print('--No attack within bounds found--')
            count += 1
            if early_stop == count:
                print('No more adversarials found ----> early stop!')
                break
            continue

        count = 0

        classes = classification(adv[0], fmodel)
        min_dim = n_adv_dims
        for i, a in enumerate(adv[0]):
            a_ = u.t2n(a.flatten())
            pert_length = np.linalg.norm(a_ - x_orig[i], ord=2)
            if run == 0:
                min_dim = 1
                advs.append(np.array([a_]))
                pert_lengths.append(np.array([pert_length]))
                adv_dirs.append(np.array([(a_ - x_orig[i]) / pert_length]))
                adv_class.append(np.array([classes[i]]))
            else:
                dim = np.sum(pert_lengths[i] < pert_length)
                min_dim = np.minimum(min_dim, dim) + 1
                advs[i] = np.vstack([advs[i][:dim], a_])
                adv_dirs[i] = np.vstack([adv_dirs[i][:dim], (a_ - x_orig[i]) / pert_length])
                adv_class[i] = np.append(adv_class[i][:dim], classes[i])
                pert_lengths[i] = np.append(pert_lengths[i][:dim], pert_length)
        dirs = dirs_to_attack_format(adv_dirs)
        if min_dim == n_adv_dims:
            break

    return advs, adv_dirs, adv_class, pert_lengths