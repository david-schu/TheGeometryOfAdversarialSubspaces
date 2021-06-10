import torch
from attacks import OrthogonalAttack, CarliniWagner
from utils import classification, dev
from tqdm import tqdm
from utils import orth_check

def run_batch(fmodel,
              images,
              labels,
              attack_params,
              random_start=True,
              input_attack=CarliniWagner,
              n_adv_dims=3,
              early_stop=3,
              epsilons=[None],
              verbose=False,
              orth_const=100
    ):

    # initialize variables
    n_pixel = images.shape[-1] ** 2
    n_images = images.shape[0]
    n_channels = images.shape[1]
    x_orig = images.reshape([n_images, n_channels, n_pixel])

    count = 0
    pert_lengths = torch.zeros((n_images, n_adv_dims), device=dev())
    adv_class = torch.zeros((n_images, n_adv_dims), device=dev(), dtype=int)
    advs = torch.zeros((n_images, n_adv_dims, n_channels, n_pixel), device=dev())
    adv_dirs = torch.zeros((n_images, n_adv_dims, n_channels, n_pixel), device=dev())
    adv_found = torch.full((n_images, n_adv_dims), False, dtype=bool, device=dev())
    dirs = torch.tensor([], device=dev())

    for run in tqdm(range(n_adv_dims), leave=False):
        if verbose:
            print('Run %d' % (run + 1))

        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
                                  random_start=random_start,
                                  orth_const=orth_const)
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
            a_ = a.flatten(-2,-1)
            pert_length = torch.norm(a_ - x_orig[i])

            advs[i, run] = a_
            adv_dirs[i, run] = (a_ - x_orig[i]) / pert_length
            adv_class[i, run] = classes[i]
            pert_lengths[i, run] = pert_length
            adv_found[i, run] = True
        dirs = adv_dirs[:, :run+1]
        print(orth_check(dirs[0]))

    return advs, adv_dirs, adv_class, pert_lengths, adv_found