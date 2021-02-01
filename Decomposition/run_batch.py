import torch
from attacks import OrthogonalAttack, CarliniWagner
from utils import classification, dirs_to_attack_format, dev

def run_batch(fmodel,
              images,
              labels,
              attack_params,
              pre_data=None,
              orth_const=50,
              random_start=True,
              input_attack=CarliniWagner,
              n_adv_dims=3,
              max_runs=100,
              early_stop=3,
              epsilons=[None],
              plot_loss=False,
              verbose=False,
              save_dims=False
    ):

    # initialize variables
    n_pixel = images.shape[-1] ** 2
    n_images = images.shape[0]
    x_orig = images.reshape([n_images, n_pixel])

    count = 0
    min_dim = 0

    pert_lengths = torch.zeros((n_images, n_adv_dims), device=dev())
    adv_class = torch.zeros((n_images, n_adv_dims), device=dev(), dtype=int)
    advs = torch.zeros((n_images, n_adv_dims, n_pixel), device=dev())
    adv_dirs = torch.zeros((n_images, n_adv_dims, n_pixel), device=dev())
    adv_found = torch.full((n_images, n_adv_dims), False, dtype=bool, device=dev())
    dirs = torch.tensor([], device=dev())

    if save_dims:
        dims = torch.tensor([], device=dev()).reshape((0, n_images))

    if not pre_data is None:
        adv_found[:, :pre_data['adv_found'].shape[-1]] = pre_data['adv_found']
        pert_lengths[:, :pre_data['adv_found'].shape[-1]] = pre_data['pert_lengths']
        adv_class[:, :pre_data['adv_found'].shape[-1]] = pre_data['adv_class']
        advs[:, :pre_data['adv_found'].shape[-1]] = pre_data['advs']
        adv_dirs[:, :pre_data['adv_found'].shape[-1]] = pre_data['adv_dirs']
        advs[~adv_found] = 0
        adv_dirs[~adv_found] = 0
        adv_class[~adv_found] = 0
        pert_lengths[~adv_found] = 0
        dirs = dirs_to_attack_format(adv_dirs)

    for run in range(max_runs):
        if verbose:
            print('Run %d - Adversarial Dimension %d...' % (run + 1, min_dim + 1))

        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
                                  orth_const=orth_const,
                                  plot_loss=plot_loss,
                                  random_start=random_start)
        _, adv, success = attack(fmodel, images, labels, epsilons=epsilons)

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
            if not success[0,i]:
                continue
            a_ = a.flatten()
            pert_length = torch.norm(a_ - x_orig[i])
            dim = torch.sum(pert_lengths[i][torch.nonzero(pert_lengths[i])] < pert_length)

            if dim >= n_adv_dims:
                continue

            advs[i, dim] = a_
            adv_dirs[i, dim] = (a_ - x_orig[i]) / pert_length
            adv_class[i, dim] = classes[i]
            pert_lengths[i, dim] = pert_length
            adv_found[i, dim] = True
            adv_found[i,dim+1:] = False

        advs[~adv_found] = 0
        adv_dirs[~adv_found] = 0
        adv_class[~adv_found] = 0
        pert_lengths[~adv_found] = 0
        if save_dims:
            dims = torch.cat([dims, torch.sum(adv_found, dim=1)], 0)

        dirs = dirs_to_attack_format(adv_dirs)
        min_dim = torch.min(torch.sum(adv_found, dim=1))
        if min_dim == n_adv_dims:
            break
    print('Runs needed for %d directions: %d' % (min_dim, run + 1 ))

    if save_dims:
        return advs, adv_dirs, adv_class, pert_lengths, adv_found, dims
    return advs, adv_dirs, adv_class, pert_lengths, adv_found