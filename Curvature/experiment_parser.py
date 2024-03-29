import sys
import os.path

from curve_utils import load_mnist, load_cifar

import numpy as np

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


def load_dictionary(filename):
    if os.path.exists(filename): 
        dictionary = np.load(filename, allow_pickle=True)['data'].item()
    else:
        assert False, (f'ERROR: get_experiment_output: File {filename} not found.')
    return dictionary


def get_run_name(run_type):
    if run_type == 0: # natural, paired
        run_name = 'natural_paired_'
    elif run_type == 1: # robust, paired
        run_name = 'robust_paired_'
    elif run_type == 2: # natural, adversarial
        run_name = 'natural_adv_'
    elif run_type == 3: # robust, adversarial
        run_name = 'robust_adv_'
    elif run_type == 4: # natural, random
        run_name = 'natural_rand_subspace_'
    elif run_type == 5: # robust, random
        run_name = 'robust_rand_subspace_'
    elif run_type == 6: # natural, adversarial
        run_name = 'natural_adv_subspace_'
    elif run_type == 7: # robust, adversarial
        run_name = 'robust_adv_subspace_'
    return run_name


def get_dataset_prefix(dataset_type):
    return 'mnist' if dataset_type == 0 else 'cifar'


def get_index_data(dataset_type, run_type, num_images, num_advs, code_directory='../../'):
    dataset_prefix = get_dataset_prefix(dataset_type)
    filename_prefix = code_directory+'AdversarialDecomposition/data/'
    filename_prefix += f'{dataset_prefix}_batch/'
    if not os.path.exists(filename_prefix):
        assert False, (f'ERROR: get_experiment_output: Directory {filename_prefix} not found.')
    all_origin_indices_filename = filename_prefix+dataset_prefix+f'_{num_images}image_indices.npz'
    all_origin_indices_dict = load_dictionary(all_origin_indices_filename)
    adv_origin_indices_filename = filename_prefix+dataset_prefix+f'_{num_images}image_{num_advs}adv_indices.npz'
    adv_origin_indices_dict = load_dictionary(adv_origin_indices_filename)
    return all_origin_indices_dict, adv_origin_indices_dict


def get_experiment_outputs(dataset_type, run_type, image_index, code_directory='../../'):
    filename_prefix = code_directory+'AdversarialDecomposition/data/'
    dataset_prefix = get_dataset_prefix(dataset_type)
    filename_prefix += f'{dataset_prefix}_batch/'
    if not os.path.exists(filename_prefix):
        assert False, (f'ERROR: get_experiment_output: Directory {filename_prefix} not found.')
    run_name = get_run_name(run_type)
    if run_type <= 3: #mean curvature calculations
        filename_postfix = dataset_prefix+f'_{image_index:03d}_curvatures_and_directions_autodiff.npz'
    else: # subspace experiments
        filename_postfix = dataset_prefix+f'_{image_index:03d}_curvatures_autodiff.npz'
    filename = filename_prefix + run_name + filename_postfix
    experiment_dict = load_dictionary(filename)
    return experiment_dict

def stack_with_padding(list_of_arrays):
    ndim = list_of_arrays[0].ndim
    shapes = np.stack([np.array(list(array.shape)) for array in list_of_arrays], axis=0)
    max_dims = np.max(shapes, axis=0, keepdims=True)
    pad_amount = max_dims - shapes
    pads = [list(pad_amount[i,:]) for i in range(len(list_of_arrays))]
    pads = [[(0, pads[i][j]) for j in range(ndim)] for i in range(len(pads))]
    new_array_list = [np.pad(array, pad, mode='empty') for pad, array in zip(pads, list_of_arrays)]
    return np.stack(new_array_list, axis=0)


def get_combined_experiment_outputs(dataset_type, run_type, num_images, code_directory='../../'):
    principal_curvatures = []
    principal_directions = []
    origin_indices = []
    for image_index in range(num_images):
        experiment_outputs = get_experiment_outputs(dataset_type, run_type, image_index, code_directory)    
        principal_curvatures.append(np.squeeze(experiment_outputs['principal_curvatures']))
        principal_directions.append(np.squeeze(experiment_outputs['principal_directions']))
        origin_indices.append(experiment_outputs['origin_indices'][0])
    principal_curvatures = stack_with_padding(principal_curvatures)
    principal_directions = stack_with_padding(principal_directions)
    origin_indices = np.stack(origin_indices, axis=0)
    output_dict = {
        'principal_curvatures':principal_curvatures,
        'principal_directions':principal_directions,
        'origin_indices':origin_indices,
    }
    return output_dict
