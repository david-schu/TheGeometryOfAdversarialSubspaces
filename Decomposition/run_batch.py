import torch
from attacks import OrthogonalAttack, CarliniWagner
from utils import classification, dev
from tqdm import tqdm
import foolbox
from utils import orth_check

def run_batch(model,
              images,
              labels,
              attack_params,
              random_start=True,
              input_attack=CarliniWagner,
              n_adv_dims=3,
              early_stop=3,
              epsilons=[None],
              verbose=False
    ):
    fmodel = foolbox.models.PyTorchModel(model,  # return logits in shape (bs, n_classes)
                                         bounds=(0., 1.),  # num_classes=10,
                                         device=dev())

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
    dirs=[]



    for run in tqdm(range(n_adv_dims), leave=False):
        if verbose:
            print('Run %d' % (run + 1))

        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
                                  random_start=random_start)

        _, adv, success = attack(fmodel, images, labels, epsilons=epsilons)
        adv = adv[0]
        # check if adversarials were found and stop early if not
        if success.sum() == 0 or (adv==0).all():
            print('--No attack within bounds found--')
            count += 1
            if early_stop == count:
                print('No more adversarials found ----> early stop!')
                break
            continue

        count = 0

        classes = classification(adv, model)
        for i, a in enumerate(adv):
            if not success[0, i] or (a==0).all():
                continue
            a_ = a.flatten(-2, -1)
            pert_length = torch.norm(a_ - x_orig[i])

            advs[i, run] = a_
            adv_dir = (a_ - x_orig[i]) / pert_length
            adv_dirs[i, run] = adv_dir
            adv_class[i, run] = classes[i]
            pert_lengths[i, run] = pert_length
            adv_found[i, run] = True

        dirs = adv_dirs[0, :run+1].flatten(-2, -1).detach().cpu().numpy()
        # print(orth_check(dirs[0]))


    return advs, adv_dirs, adv_class, pert_lengths, adv_found