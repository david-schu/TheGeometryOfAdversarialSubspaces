import sys
import os.path

from tqdm import tqdm
import numpy as np
import torch

sys.path.insert(0, './..')
sys.path.insert(0, '../data')

from utils import dev
from curve_utils import *

sys.path.insert(0, './../..')

import response_contour_analysis.utils.dataset_generation as data_utils
import response_contour_analysis.utils.model_handling as model_utils
import response_contour_analysis.utils.principal_curvature as curve_utils

if __name__ == "__main__":
    """
    dataset_type
        0 - MNIST
        1 - CIFAR
    run_type
        0 - natural, paired boundary
        1 - robust, paired boundary

        2 - natural, adversarial boundary
        3 - robust, adversarial boundary

        4 - natural, random subspace
        5 - robust, random subspace

        6 - natural, adversarial subspace
        7 - robust, adversarial subspace
    """
    dataset_type = int(sys.argv[1])
    run_type = int(sys.argv[2])
    image_index = int(sys.argv[3])

    code_directory = '../../'
    filename_prefix = code_directory+'AdversarialDecomposition/data/'

    batch_size = 10
    num_images = 2#50
    num_advs = 2#10
    seed = 0

    num_iters = 2 # for paired image boundary search
    num_steps_per_iter = 100 # for paired image boundary search
    dtype = torch.double

    if dataset_type == 0: # MNIST
        model_natural, data_natural, model_robust, data_robust = load_mnist(code_directory, seed)
        data_prefix = 'mnist'
    else: # CIFAR
        model_natural, data_natural, model_robust, data_robust = load_cifar(code_directory)
        data_prefix = 'cifar'

    # valid_indices is a list of all indices where the model output matches the data label. This should be every index.
    # origin_indices is a random subset of valid_indices based on the num_images parameter
    image_index_filename = filename_prefix+data_prefix+f'_{num_images}image_indices.npz'
    if os.path.exists(image_index_filename): 
        index_dict = np.load(image_index_filename, allow_pickle=True)['data'].item()
        all_natural_origin_indices = index_dict['natural_origin']
        all_natural_valid_indices = index_dict['natural_valid']
        all_robust_origin_indices = index_dict['robust_origin']
        all_robust_valid_indices = index_dict['robust_valid']
    else:
        all_natural_origin_indices, all_natural_valid_indices = get_origin_indices(model_natural, data_natural, num_images, num_advs=None)
        all_robust_origin_indices, all_robust_valid_indices = get_origin_indices(model_robust, data_robust, num_images, num_advs=None)
        np.savez(image_index_filename, data={
            'natural_origin':all_natural_origin_indices,
            'natural_valid':all_natural_valid_indices,
            'robust_origin':all_robust_origin_indices,
            'robust_valid':all_robust_valid_indices
        })
        
    print('Data and models loaded')

    if run_type <= 3: #mean curvature calculations
        # need to make a new subset of indices that also have enough valid adversarial examples
        adv_image_index_filename = filename_prefix+data_prefix+f'_{num_images}image_{num_advs}adv_indices.npz'
        if os.path.exists(adv_image_index_filename): 
            index_dict = np.load(adv_image_index_filename, allow_pickle=True)['data'].item()
            data_natural_paired = index_dict['data_natural_paired']
            data_robust_paired = index_dict['data_robust_paired']
            paired_natural_adv_origin_indices = index_dict['paired_natural_adv_origin']
            paired_robust_adv_origin_indices = index_dict['paired_robust_adv_origin']
            natural_adv_origin_indices = index_dict['natural_adv_origin']
            robust_adv_origin_indices = index_dict['robust_adv_origin']
        else:
            data_natural_paired = generate_paired_dict(model_natural, data_natural, all_natural_origin_indices,
                                         all_natural_valid_indices, num_images, num_advs, num_steps_per_iter, num_iters)
            data_robust_paired = generate_paired_dict(model_robust, data_robust, all_robust_origin_indices,
                                         all_robust_valid_indices, num_images, num_advs, num_steps_per_iter, num_iters)
            paired_natural_adv_origin_indices = get_origin_indices(model_natural, data_natural_paired, num_images, num_advs)[0]
            paired_robust_adv_origin_indices = get_origin_indices(model_robust, data_robust_paired, num_images, num_advs)[0]
            natural_adv_origin_indices = get_origin_indices(model_natural, data_natural, num_images, num_advs)[0]
            robust_adv_origin_indices = get_origin_indices(model_robust, data_robust, num_images, num_advs)[0]
            np.savez(adv_image_index_filename, data={
                'data_natural_paired':data_natural_paired,
                'data_robust_paired':data_robust_paired,
                'paired_natural_adv_origin':paired_natural_adv_origin_indices,
                'paired_robust_adv_origin':paired_robust_adv_origin_indices,
                'natural_adv_origin':natural_adv_origin_indices,
                'robust_adv_origin':robust_adv_origin_indices
            })

        filename_postfix = data_prefix+'_curvatures_and_directions_autodiff.npz'
        if run_type == 0: # natural, paired
            data_ = data_natural_paired
            model_ = model_natural
            run_name = 'natural_paired_'

        elif run_type == 1: # robust, paired
            data_ = data_robust_paired
            model_ = model_robust
            run_name = 'robust_paired_'

        elif run_type == 2: # natural, adversarial
            data_ = data_natural
            model_ = model_natural
            run_name = 'natural_adv_'

        elif run_type == 3: # robust, adversarial
            data_ = data_robust
            model_ = model_robust
            run_name = 'robust_adv_'
        print('experiment ' + run_name)

        if run_type == 0: # natural, paired condition
            condition_origin_indices = paired_natural_adv_origin_indices
        elif run_type == 1: # robust, paired condition
            condition_origin_indices = paired_robust_adv_origin_indices
        elif run_type == 2: # natural, adversarial condition
            condition_origin_indices = natural_adv_origin_indices
        elif run_type == 3: # robust, adversarial condition
            condition_origin_indices = robust_adv_origin_indices

        condition_zip = zip([model_], [data_])
        shape_operators, principal_curvatures, principal_directions = get_curvature(
            condition_zip, [condition_origin_indices[image_index]], num_advs, num_iters, num_steps_per_iter, dtype)

        save_dict = {}
        if run_type <= 1: # paired conditions
            save_dict['data'] = data_
        else: # adversarial conditions
            save_dict['data'] = []
        save_dict['origin_indices'] = [condition_origin_indices[image_index]]
        save_dict['shape_operators'] = shape_operators
        save_dict['principal_curvatures'] = principal_curvatures
        save_dict['principal_directions'] = principal_directions
        save_dict['mean_curvatures'] = np.mean(principal_curvatures, axis=-1)


    else: # subspace experiments
        filename_postfix = data_prefix+'_curvatures_autodiff.npz'

        if run_type == 4: # natural, random
            origin_indices = [all_natural_origin_indices[image_index]]
            model_ = model_natural
            data_ = data_natural
            run_name = 'natural_rand_subspace_'
        elif run_type == 5: # robust, random
            origin_indices = [all_robust_origin_indices[image_index]]
            model_ = model_robust
            data_ = data_robust
            run_name = 'robust_rand_subspace_'
        elif run_type == 6: # natural, adversarial
            origin_indices = [all_natural_origin_indices[image_index]]
            model_ = model_natural
            data_ = data_natural
            run_name = 'natural_adv_subspace_'
        elif run_type == 7: # robust, adversarial
            origin_indices = [all_robust_origin_indices[image_index]]
            model_ = model_robust
            data_ = data_robust
            run_name = 'robust_adv_subspace_'
        print('experiment ' + run_name)

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
                #hessian = torch.autograd.functional.hessian(func, torchify(boundary_image[None,...]))
                hessian = torch.eye(int(boundary_image.size)).type(dtype).to(dev())
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