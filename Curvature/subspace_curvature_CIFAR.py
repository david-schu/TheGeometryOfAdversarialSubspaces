import sys

import dill
from tqdm import tqdm
import numpy as np
import torch
from torchvision import datasets

sys.path.insert(0, './..')
sys.path.insert(0, '../data')

from models import model as model_loader
from utils import dev, make_orth_basis
from robustness1.datasets import CIFAR
from curve_utils import *

sys.path.insert(0, './../..')

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
    run_type = int(sys.argv[1])

    code_directory = '/home/bethge/dpaiton/Work/'
    filename_prefix = code_directory+'AdversarialDecomposition/data/'
    num_iters = 2 # for paired image boundary search
    num_steps_per_iter = 100 # for paired image boundary search
    buffer_portion = 0.25
    num_eps = 1000

    batch_size = 10
    num_images = 50
    num_advs = 10

    # load data
    data_natural = np.load(code_directory+'AdversarialDecomposition/data/cifar_natural_diff.npy', allow_pickle=True).item()
    data_madry = np.load(code_directory+'AdversarialDecomposition/data/cifar_robust_diff.npy', allow_pickle=True).item()

    # load models
    ds = CIFAR(code_directory+'AdversarialDecomposition/data/cifar-10-batches-py')
    classifier_model = ds.get_model('resnet50', False)

    model_natural = model_loader.cifar_pretrained(classifier_model, ds)
    resume_path = code_directory+'AdversarialDecomposition/models/nat_diff.pt'
    checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))
    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'
    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model_natural.load_state_dict(sd)
    model_natural.to(dev())
    model_natural.double()
    model_natural.eval()

    model_madry = model_loader.cifar_pretrained(classifier_model, ds)
    resume_path = code_directory+'AdversarialDecomposition/models/rob_diff.pt'
    checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))
    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'
    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]:v for k,v in sd.items()}
    model_madry.load_state_dict(sd)
    model_madry.to(dev())
    model_madry.double()
    model_madry.eval()

    print('Data and models loaded.')

    if run_type <= 3: #mean curvature calculations
        filename_postfix = 'cifar_curvatures_and_directions_autodiff.npz'

        if run_type == 0: # natural, paired
          data_ = generate_paired_dict(data_natural, model_natural, num_images, num_advs)
          model_ = model_natural
          run_name = 'mean_paired_natural_'
        elif run_type == 1: # madry, paired
          data_ = generate_paired_dict(data_madry, model_madry, num_images, num_advs)
          model_ = model_madry
          run_name = 'mean_paired_madry_'
        elif run_type == 2: # natural, adversarial
          data_ = data_natural
          model_ = model_natural
          run_name = 'mean_adv_natural_'
        elif run_type == 3: # madry, adversarial
          data_ = data_madry
          model_ = model_madry
          run_name = 'mean_adv_madry_'

        origin_indices = get_origin_indices(model_, data_, num_images)
        condition_zip = zip([model_], [data_])
        shape_operators, principal_curvatures, principal_directions = get_curvature(
            condition_zip, num_images, num_advs, num_eps, batch_size, buffer_portion,
            origin_indices, autodiff=True)
        mean_curvatures = np.mean(principal_curvatures, axis=-1)

        save_dict = {}
        if run_type <= 1: # paired conditions
          save_dict[run_name+'data'] = data_
        else: # adversarial conditions
          save_dict[run_name+'data'] = []
        save_dict[run_name+'shape_operators'] = shape_operators
        save_dict[run_name+'principal_curvatures'] = principal_curvatures
        save_dict[run_name+'principal_directions'] = principal_directions
        save_dict[run_name+'mean_curvatures'] = mean_curvatures
        save_dict[run_name+'origin_indices'] = origin_indices


    else: # subspace experiments
        filename_postfix = 'cifar_subspace_curvatures_autodiff.npz'

        if run_type == 4: # natural, random
            model_ = model_natural
            data_ = data_natural
            run_name = 'subspace_rand_natural_'
        elif run_type == 5: # madry, random
            model_ = model_madry
            data_ = data_madry
            run_name = 'subspace_rand_madry'
        elif run_type == 6: # natural, adversarial
            model_ = model_natural
            data_ = data_natural
            run_name = 'subspace_adv_natural'
        elif run_type == 7: # madry, adversarial
            model_ = model_madry
            data_ = data_madry
            run_name = 'subspace_adv_madry'

        origin_indices = get_origin_indices(model_, data_, num_images)

        dtype = torch.double
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
                def func(x):
                    acts_diff = paired_activation(model_, x, clean_lbl, adv_lbl)
                    return acts_diff
                hessian = torch.autograd.functional.hessian(func, torchify(boundary_image[None,...]))
                hessian = hessian.reshape((int(boundary_image.size), int(boundary_image.size))).type(dtype)

                if run_type == 4 or run_type == 5: # random subspace
                    dirs = [(gradient / torch.linalg.norm(gradient)).detach().cpu().numpy()]
                    n_pixels = gradient.numel()
                    n_iterations = 3
                    random_basis = torch.from_numpy(make_orth_basis(dirs, n_pixels, n_iterations)[:subspace_size, :]).type(dtype)
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
                    adv_basis = torch.from_numpy(adv_dirs).type(dtype)
                    curvature = curve_utils.local_response_curvature_isoresponse_surface(gradient, hessian, projection_subspace_of_interest=adv_basis)
                    adv_subspace_curvatures = curvature[1].detach().cpu().numpy()
                    all_subspace_curvatures[image_idx, :] = adv_subspace_curvatures
                pbar.update(1)
        pbar.close()

        save_dict = {}
        save_dict[run_name+'origin_indices'] = origin_indices
        save_dict[run_name+'principal_curvatures'] = all_subspace_curvatures

    np.savez(filename_prefix+run_name+filename_postfix, data=save_dict)
