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
    pert_lengths = np.zeros((n_images, n_adv_dims))
    adv_class = np.zeros((n_images, n_adv_dims))
    advs = np.zeros((n_images, n_adv_dims, n_pixel))
    adv_dirs = np.zeros((n_images, n_adv_dims, n_pixel))
    dirs = torch.tensor([])
    adv_found = np.full((n_images, n_adv_dims), False, dtype=bool)

    for run in range(max_runs):
        print('Run %d - Adversarial Dimension %d...' % (run + 1, min_dim + 1))

        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
                                  orth_const=orth_const,
                                  plot_loss=plot_loss)
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
        for i, a in enumerate(adv[0]):
            if not success[0][i]:
                continue
            a_ = u.t2n(a.flatten())
            pert_length = np.linalg.norm(a_ - x_orig[i], ord=2)
            dim = np.sum(pert_lengths[i][np.nonzero(pert_lengths[i])] < pert_length)

            if dim >= n_adv_dims:
                continue

            advs[i, dim] = a_
            adv_dirs[i, dim] = (a_ - x_orig[i]) / pert_length
            adv_class[i, dim] = classes[i]
            pert_lengths[i, dim] = pert_length
            adv_found[i, dim] = True
            adv_found[i,dim+1:] = False

        advs[~adv_found] = adv_dirs[~adv_found] = adv_class[~adv_found] = pert_lengths[~adv_found] = 0

        dirs = dirs_to_attack_format(adv_dirs)
        min_dim = np.amax(np.sum(adv_found, axis=1))
        if min_dim == n_adv_dims:
            break

    return advs, adv_dirs, adv_class, pert_lengths