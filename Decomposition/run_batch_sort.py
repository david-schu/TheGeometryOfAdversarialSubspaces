import torch
from attacks import OrthogonalAttack, CarliniWagner
from utils import classification, dirs_to_attack_format, dev

def run_batch(fmodel,
              images,
              labels,
              attack_params,
              pre_data=None,
              random_start=True,
              input_attack=CarliniWagner,
              n_adv_dims=3,
              max_runs=100,
              early_stop=3,
              epsilons=[None],
              plot_loss=False,
              verbose=False
    ):

    # initialize variables
    n_pixel = images.shape[-1] ** 2
    n_images = images.shape[0]
    x_orig = images.reshape([n_images, n_pixel])

    count = 0
    n_adv_dims = max_runs
    pert_lengths = torch.zeros((n_images, n_adv_dims), device=dev())
    adv_class = torch.zeros((n_images, n_adv_dims), device=dev(), dtype=int)
    advs = torch.zeros((n_images, n_adv_dims, n_pixel), device=dev())
    adv_dirs = torch.zeros((n_images, n_adv_dims, n_pixel), device=dev())
    adv_found = torch.full((n_images, n_adv_dims), False, dtype=bool, device=dev())
    dirs = torch.tensor([], device=dev())

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
            print('Run %d' % (run + 1))

        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
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
            if not success[0, i]:
                continue
            a_ = a.flatten()
            pert_length = torch.norm(a_ - x_orig[i])

            advs[i, run] = a_
            adv_dirs[i, run] = (a_ - x_orig[i]) / pert_length
            adv_class[i, run] = classes[i]
            pert_lengths[i, run] = pert_length
            adv_found[i, run] = True

        dirs = dirs_to_attack_format(adv_dirs)

    return advs, adv_dirs, adv_class, pert_lengths, adv_found