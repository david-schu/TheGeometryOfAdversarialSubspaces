import torch
from attacks import OrthogonalAttack, L2OrthAttack
from utils import classification, dev
import foolbox
import numpy as np ###

def run_attack(model,
               image,
               label,
               attack_params,
               save_path,
               random_start=True,
               input_attack=L2OrthAttack,
               n_adv_dims=3,
               early_stop=3,
               epsilons=[None],
               verbose=False
    ):
    fmodel = foolbox.models.PyTorchModel(model,  # return logits in shape (bs, n_classes)
                                         bounds=(0., 1.),  # num_classes=10,
                                         device=dev())

    # initialize variables
    n_pixel = image.shape[-1] ** 2
    n_channels = image.shape[1]
    x_orig = image.flatten()

    count = 0
    pert_lengths = torch.zeros(n_adv_dims, device=dev())
    adv_class = torch.zeros(n_adv_dims, device=dev(), dtype=int)
    advs = torch.zeros((n_adv_dims, n_channels * n_pixel), device=dev())
    adv_dirs = torch.zeros((n_adv_dims, n_channels * n_pixel), device=dev())
    dirs=[]

    dim = 0
    run = 0
    while dim < n_adv_dims:

        if verbose:
            print('Run %d' % (run + 1))
        run += 1
        attack = OrthogonalAttack(input_attack=input_attack,
                                  params=attack_params,
                                  adv_dirs=dirs,
                                  random_start=random_start)
        _, adv, success = attack(fmodel, image, label, epsilons=epsilons)
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

        class_ = classification(adv, model, orig_label=label,is_adv=True)
        a_ = adv.flatten()
        pert_length = torch.norm(a_ - x_orig)

        advs[dim] = a_
        adv_dir = (a_ - x_orig) / pert_length
        adv_dirs[dim] = adv_dir
        adv_class[dim] = class_
        pert_lengths[dim] = pert_length

        dirs = adv_dirs[:dim+1].detach().cpu().numpy()
        dim += 1
        
        ###
        data = {
            'advs': advs.unsqueeze(0).cpu().detach().numpy(),
            'dirs': adv_dirs.unsqueeze(0).cpu().detach().numpy(),
            'adv_class': adv_class.unsqueeze(0).cpu().detach().numpy(),
            'pert_lengths': pert_lengths.unsqueeze(0).cpu().detach().numpy(),
            'images': image.detach().cpu().numpy(),
            'labels': label.detach().cpu().numpy(),
        }

        np.save(save_path, data)
        ####

    print('Dimensions' + str(dim))
    return advs, adv_dirs, adv_class, pert_lengths