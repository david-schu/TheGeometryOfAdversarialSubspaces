import sys

#import dill
from tqdm import tqdm
import numpy as np
import torch

sys.path.insert(0, './..')
sys.path.insert(0, '../data')

#from models import model as model_loader
from utils import dev
#from robustness1.datasets import CIFAR
from curve_utils import *

sys.path.insert(0, './../..')

import response_contour_analysis.utils.dataset_generation as data_utils
import response_contour_analysis.utils.model_handling as model_utils
import response_contour_analysis.utils.principal_curvature as curve_utils

if __name__ == "__main__":
    """
    run_type
        0 - natural, paired boundary
        1 - madry, paired boundary

        2 - natural, adversarial boundary
        3 - madry, adversarial boundary

        4 - natural, random subspace
        5 - madry, random subspace

        6 - natural, adversarial subspace
        7 - madry, adversarial subspace
    """
    dataset_type = int(sys.argv[1])
    run_type = int(sys.argv[2])

    code_directory = '../../'
    filename_prefix = code_directory+'AdversarialDecomposition/data/'

    batch_size = 10
    num_images = 50
    num_advs = 10
    seed = 0

    num_iters = 2 # for paired image boundary search
    num_steps_per_iter = 100 # for paired image boundary search
    dtype = torch.double

    if dataset_type == 0: # MNIST
        model_natural, data_natural, model_madry, data_madry = load_mnist(code_directory, seed)
        data_prefix = 'mnist'
    else: # CIFAR
        model_natural, data_natural, model_madry, data_madry = load_cifar(code_directory)
        data_prefix = 'cifar'

    print('Data and models loaded')

    if run_type <= 3: #mean curvature calculations
        filename_postfix = data_prefix+'_curvatures_and_directions_autodiff.npz'

        if run_type == 0: # natural, paired
            origin_indices, valid_indices = get_origin_indices(model_natural, data_natural, num_images, num_advs=None)
            data_ = generate_paired_dict(model_natural, data_natural, origin_indices,
                                         valid_indices, num_images, num_advs, num_steps_per_iter, num_iters)
            model_ = model_natural
            run_name = 'natural_paired_'

        elif run_type == 1: # madry, paired
            origin_indices, valid_indices = get_origin_indices(model_madry, data_madry, num_images, num_advs=None)
            data_ = generate_paired_dict(model_madry, data_madry, origin_indices,
                                         valid_indices, num_images, num_advs, num_steps_per_iter, num_iters)
            model_ = model_madry
            run_name = 'madry_paired_'

        elif run_type == 2: # natural, adversarial
            data_ = data_natural
            model_ = model_natural
            run_name = 'natural_adv_'

        elif run_type == 3: # madry, adversarial
            data_ = data_madry
            model_ = model_madry
            run_name = 'madry_adv_'
        print('experiment ' + run_name)

        condition_zip = zip([model_], [data_])
        condition_origin_indices, condition_valid_indices = get_origin_indices(model_, data_, num_images, num_advs)
        shape_operators, principal_curvatures, principal_directions = get_curvature(
            condition_zip, num_images, num_advs, condition_origin_indices, num_iters, num_steps_per_iter, dtype)

        save_dict = {}
        if run_type <= 1: # paired conditions
            save_dict['data'] = data_
        else: # adversarial conditions
            save_dict['data'] = []
        save_dict['origin_indices'] = condition_origin_indices
        save_dict['shape_operators'] = shape_operators
        save_dict['principal_curvatures'] = principal_curvatures
        save_dict['principal_directions'] = principal_directions
        save_dict['mean_curvatures'] = np.mean(principal_curvatures, axis=-1)


    else: # subspace experiments
        filename_postfix = data_prefix+'_subspace_curvatures_autodiff.npz'

        if run_type == 4: # natural, random
            model_ = model_natural
            data_ = data_natural
            run_name = 'natural_rand_'
        elif run_type == 5: # madry, random
            model_ = model_madry
            data_ = data_madry
            run_name = 'madry_rand_'
        elif run_type == 6: # natural, adversarial
            model_ = model_natural
            data_ = data_natural
            run_name = 'natural_adv_'
        elif run_type == 7: # madry, adversarial
            model_ = model_madry
            data_ = data_madry
            run_name = 'madry_adv_'
        print('experiment ' + run_name)

        origin_indices = get_origin_indices(model_, data_, num_images, num_advs=None)[0]

        subspace_size = num_advs-1
        all_subspace_curvatures = np.zeros((num_images, subspace_size))
        pbar = tqdm(total=num_advs*num_images, leave=False)
        for image_idx, origin_idx in enumerate(list(origin_indices)):
            for adv_idx in range(num_advs):
                boundary_image, boundary_dir, pert_length = get_paired_boundary_image(
                    model=model_,
                    origin=data_['images'][origin_idx, ...],
                    alt_image=data_['advs'][origin_idx, adv_idx, ...],
                    num_steps_per_iter=num_steps_per_iter,
                    num_iters=num_iters)
                clean_lbl = int(data_['labels'][origin_idx])
                adv_lbl = int(data_['adv_class'][origin_idx, adv_idx])
                activation, gradient = paired_activation_and_gradient(model_,
                        torchify(boundary_image[None, ...]), clean_lbl, adv_lbl)
                gradient = gradient.reshape(-1).type(dtype)
                n_pixels = gradient.numel()
                def func(x):
                    acts_diff = paired_activation(model_, x, clean_lbl, adv_lbl)
                    return acts_diff
                hessian = torch.autograd.functional.hessian(func, torchify(boundary_image[None,...]))
                hessian = hessian.reshape((int(boundary_image.size), int(boundary_image.size))).type(dtype)

                if run_type == 4 or run_type == 5: # random subspace
                    norm_gradient = (gradient / torch.linalg.norm(gradient)).detach().cpu().numpy()
                    random_basis = torch.from_numpy(data_utils.get_rand_orth_vectors(norm_gradient, num_orth_directions=subspace_size)).type(dtype).to(dev())
                    curvature = curve_utils.local_response_curvature_isoresponse_surface(gradient, hessian, projection_subspace_of_interest=random_basis)
                    rand_subspace_curvatures = curvature[1].detach().cpu().numpy()
                    all_subspace_curvatures[image_idx, :] = rand_subspace_curvatures

                elif run_type == 6 or run_type == 7: # adversarial subspace
                    adv_dirs = data_['dirs'][origin_idx, :num_advs, ...].reshape(num_advs, n_pixels)
                    if adv_idx > 0 and adv_idx < num_advs: # exclude current perturbation direction
                        adv_dirs = np.concatenate((adv_dirs[:adv_idx], adv_dirs[adv_idx+1:]))
                    elif adv_idx == 0:
                        adv_dirs = adv_dirs[adv_idx+1:]
                    else:
                        adv_dirs = adv_dirs[:adv_idx]
                    adv_basis = torch.from_numpy(adv_dirs).type(dtype).to(dev())
                    curvature = curve_utils.local_response_curvature_isoresponse_surface(gradient, hessian, projection_subspace_of_interest=adv_basis)
                    adv_subspace_curvatures = curvature[1].detach().cpu().numpy()
                    all_subspace_curvatures[image_idx, :] = adv_subspace_curvatures
                pbar.update(1)
        pbar.close()

        save_dict = {}
        save_dict['origin_indices'] = origin_indices
        save_dict['principal_curvatures'] = all_subspace_curvatures

    filename = filename_prefix + run_name + filename_postfix
    np.savez(filename, data=save_dict)
    print(f'output saved to {filename}')