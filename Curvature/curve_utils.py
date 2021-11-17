import sys

import numpy as np
import torch
import dill
from tqdm import tqdm

sys.path.insert(0, './..')
sys.path.insert(0, '../data')

from models import model as model_loader
from utils import dev
from robustness1.datasets import CIFAR
from models.cifar_models import model_zoo

sys.path.insert(0, './../..')

import response_contour_analysis.utils.model_handling as model_utils
import response_contour_analysis.utils.principal_curvature as curve_utils

def tab_name_to_hex(tab):
    conv_table = {
        "tab:blue": "#1f77b4",
        "tab:orange": "#ff7f0e",
        "tab:green": "#2ca02c",
        "tab:red": "#d62728",
        "tab:purple": "#9467bd",
        "tab:brown": "#8c564b",
        "tab:pink": "#e377c2",
        "tab:gray": "#7f7f7f",
        "tab:grey": "#7f7f7f",
        "tab:olive": "#bcbd22",
        "tab:cyan": "#17becf",
    }
    return conv_table[tab.lower()]


def torchify(img):
    output = torch.from_numpy(img).type(torch.DoubleTensor).to(dev())
    output.requires_grad = True
    return output


def get_paired_boundary_image(model, origin, alt_image, num_steps_per_iter, num_iters, batch_size=None):
    input_shape = (1,)+origin.shape
    def find_pert(image_line):
        labels = get_batch_predictions(model, image_line.reshape((-1,)+origin.shape), batch_size)
        pert_lbl = correct_lbl = labels[0]
        step_idx = 1 # already know the first one
        while pert_lbl == correct_lbl:
            pert_image = image_line[step_idx, ...]
            pert_lbl = labels[step_idx]
            step_idx += 1
        return step_idx-1, pert_image
    image_line = np.linspace(origin.reshape(-1), alt_image.reshape(-1), num_steps_per_iter)
    for search_iter in range(num_iters):
        step_idx, pert_image = find_pert(image_line)
        image_line = np.linspace(image_line[step_idx - 1, ...], image_line[step_idx, ...], num_steps_per_iter)
    delta_image = origin.reshape(-1) - pert_image
    pert_length = np.linalg.norm(delta_image)
    direction = delta_image / pert_length
    return pert_image.reshape(origin.shape), direction, pert_length


def get_valid_indices(model_predictions, data, num_advs=None):
    valid_indices = [] # Need to ensure that all images are correctly labeled & have valid adversarial examples
    for image_idx in range(data['images'].shape[0]):
        if model_predictions[image_idx] == data['labels'][image_idx]: # correctly labeled
            if num_advs is None: # don't care if it has enough advs
                valid_indices.append(image_idx)
            else: # valid must also have enough adversarial examples
                if np.all(np.isfinite(data['pert_lengths'][image_idx, :num_advs])): # enough adversaries found
                    valid_indices.append(image_idx)
    return  valid_indices


def get_batch_predictions(model, data, batch_size=None):
    if batch_size is None:
        batch_size = int(np.minimum(10, data.size))
    image_splits = torch.split(torchify(data), batch_size, dim=0)
    model_predictions = []
    for batch in image_splits:
        with torch.no_grad():
            model_predictions.append(torch.argmax(model(batch), dim=1).detach().cpu().numpy())
    model_predictions = np.concatenate(model_predictions, axis=0).reshape(-1)
    return model_predictions


def get_origin_indices(model, data, num_images, num_advs=None, batch_size=None):
    if batch_size is None:
        batch_size = int(np.minimum(10, num_images))
    model_predictions = get_batch_predictions(model, data['images'], batch_size)
    valid_indices = get_valid_indices(model_predictions, data, num_advs)
    #origin_indices = np.random.choice(valid_indices, size=num_images, replace=False)
    step_size = len(valid_indices) // num_images
    last = num_images * step_size
    origin_indices = [valid_indices[i] for i in range(0, last, step_size)]
    return origin_indices, valid_indices


def generate_paired_dict(model, data, origin_indices, valid_indices, num_images, num_advs, num_steps_per_iter, num_iters):
    image_shape = data['images'].shape[1:]
    num_pixels = int(np.prod(image_shape))
    images = np.zeros((num_images,) + image_shape)
    labels = np.zeros((num_images), dtype=np.int)
    dirs = np.zeros((num_images, num_advs, 1, num_pixels))
    advs = np.zeros((num_images, num_advs, 1, num_pixels))
    pert_lengths = np.zeros((num_images, num_advs))
    adv_class = np.zeros((num_images, num_advs))
    pbar = tqdm(total=len(list(origin_indices))*num_advs, leave=True)
    for image_idx, origin_idx in enumerate(list(origin_indices)):
        images[image_idx, ...] = data['images'][origin_idx, ...]
        labels[image_idx] = data['labels'][origin_idx]
        shuffled_valid_indices = np.random.choice(valid_indices, size=len(valid_indices), replace=False)
        alt_indices = [idx for idx, alt_class in zip(shuffled_valid_indices, data['labels'][shuffled_valid_indices]) if alt_class != labels[image_idx]]
        alt_images = data['images'][alt_indices[:num_advs]]
        for dir_idx, alt_idx in enumerate(alt_indices[:num_advs]):
            alt_image = data['images'][alt_idx, ...]
            boundary_image, boundary_dir, pert_length = get_paired_boundary_image(
                model, images[image_idx, ...], alt_image,
                num_steps_per_iter=num_steps_per_iter, num_iters=num_iters)
            dirs[image_idx, dir_idx, ...] = boundary_dir.reshape(1, -1)
            advs[image_idx, dir_idx, ...] = boundary_image.reshape(1, -1)
            adv_class[image_idx,  dir_idx] = torch.argmax(model(torchify(boundary_image[None, ...]))).item()
            pert_lengths[image_idx, dir_idx] =  pert_length
            pbar.update(1)
    pbar.close()
    output_dict = {}
    output_dict['images'] = images
    output_dict['labels'] = labels
    output_dict['dirs'] = dirs
    output_dict['advs'] = advs
    output_dict['adv_class'] = adv_class
    output_dict['pert_lengths'] = pert_lengths
    return output_dict


def paired_activation(model, images, neuron1, neuron2):
    if not images.requires_grad:
        images.requires_grad = True
    model.zero_grad()
    activation1 = model_utils.unit_activation(model, images, neuron1, compute_grad=True)
    activation2 = model_utils.unit_activation(model, images, neuron2, compute_grad=True)
    activation_difference = activation1 - activation2
    return activation_difference


def paired_activation_and_gradient(model, images, neuron1, neuron2):
    activation_difference = paired_activation(model, images, neuron1, neuron2)
    grad = torch.autograd.grad(activation_difference, images)[0]
    return activation_difference, grad


def get_curvature(condition_zip, origin_indices, num_advs, num_iters, num_steps_per_iter, dtype):
    """
    A note on the gradient of the difference in activations:
    The gradient points in the direction of the origin from the boundary image.
    Therefore, for large enough eps, origin - eps * grad/|grad| will reach the boundary; and boundary + eps * grad/|grad| will reach the origin 
    """
    num_images = len(origin_indices)
    models, model_data = zip(*condition_zip)
    num_models = len(models)
    image_shape = model_data[0]['images'][0, ...][None, ...].shape
    image_size = np.prod(image_shape)
    num_dims = image_size - 1 #removes normal direction
    shape_operators = np.zeros((num_models, num_images, num_advs, num_dims, num_dims))
    principal_curvatures = np.zeros((num_models, num_images, num_advs, num_dims))
    principal_directions = np.zeros((num_models, num_images, num_advs, image_size, num_dims))
    for model_idx, (model_, data_)  in enumerate(zip(models, model_data)):
        pbar = tqdm(total=num_advs*num_images, leave=True)
        for image_idx, origin_idx in enumerate(list(origin_indices)):
            clean_lbl = int(data_['labels'][origin_idx])
            for adv_idx in range(num_advs):
                boundary_image = get_paired_boundary_image(
                    model=model_,
                    origin=data_['images'][origin_idx, ...],
                    alt_image=data_['advs'][origin_idx, adv_idx, ...],
                    num_steps_per_iter=num_steps_per_iter,
                    num_iters=num_iters
                )[0]
                adv_lbl = int(data_['adv_class'][origin_idx, adv_idx])
                def func(x):
                    acts_diff = paired_activation(model_, x, clean_lbl, adv_lbl)
                    return acts_diff
                hessian = torch.autograd.functional.hessian(func, torchify(boundary_image[None,...]))
                hessian = hessian.reshape((int(boundary_image.size), int(boundary_image.size))).type(dtype)
                activation, gradient = paired_activation_and_gradient(model_, torchify(boundary_image[None, ...]), clean_lbl, adv_lbl)
                gradient = gradient.reshape(-1).type(dtype)
                curvature = curve_utils.local_response_curvature_level_set(gradient, hessian)
                shape_operators[model_idx, image_idx, adv_idx, ...] = curvature[0].detach().cpu().numpy()
                principal_curvatures[model_idx, image_idx, adv_idx, :] = curvature[1].detach().cpu().numpy()
                principal_directions[model_idx, image_idx, adv_idx, ...] = curvature[2].detach().cpu().numpy()
                pbar.update(1)
    pbar.close()
    return shape_operators, principal_curvatures, principal_directions


def get_hessian_error(model, origin, clean_lbl, adv_lbl, abscissa, ordinate, hess_params):
    def act_func(x):
        acts_diff = paired_activation(model, x, clean_lbl, adv_lbl)
        return acts_diff
    def act_grad_func(x):
        acts_diff, grad = paired_activation_and_gradient(model, x, clean_lbl, adv_lbl)
        return acts_diff, grad
    origin.requires_grad = True
    sr1_hessian = curve_utils.sr1_hessian(
        act_grad_func, origin,
        distance=hess_params['hessian_dist'],
        n_points=hess_params['hessian_num_pts'],
        random_walk=hess_params['hessian_random_walk'],
        learning_rate=hess_params['hessian_lr'],
        return_points=False,
        progress=True)
    autodiff_hessian = torch.autograd.functional.hessian(act_func, origin)
    autodiff_hessian = autodiff_hessian.reshape((int(origin.numel()), int(origin.numel())))
    n_x_samples = 10
    n_y_samples = 100
    x = np.linspace(-hess_params['hessian_dist']/2, hess_params['hessian_dist']/2, n_x_samples)
    y = np.linspace(-hess_params['hessian_dist']*1.25, hess_params['hessian_dist']*1.25, n_y_samples)
    X, Y = np.meshgrid(x, y)
    samples = (abscissa * X.reshape((-1, 1)) + ordinate * Y.reshape((-1, 1))).reshape((-1,) + origin.shape[1:])
    samples = origin + torchify(samples)
    exact_response = act_func(samples)
    sr1_approx_response = curve_utils.hessian_approximate_response(act_grad_func, samples, sr1_hessian)
    autodiff_approx_response = curve_utils.hessian_approximate_response(act_grad_func, samples, autodiff_hessian)
    sr1_total_error = (exact_response - sr1_approx_response)
    autodiff_total_error = (exact_response - autodiff_approx_response)
    sr1_rms_error = np.sqrt(np.mean(np.square(sr1_total_error.detach().cpu().numpy())))
    autodiff_rms_error = np.sqrt(np.mean(np.square(autodiff_total_error.detach().cpu().numpy())))
    return sr1_rms_error, autodiff_rms_error


def load_mnist(code_directory, seed):
    # load data
    data_natural = np.load(code_directory+f'AdversarialDecomposition/data/natural_{seed}.npy', allow_pickle=True).item()
    data_madry = np.load(code_directory+f'AdversarialDecomposition/data/robust_{seed}.npy', allow_pickle=True).item()
    # load models
    model_natural = model_loader.madry_diff()
    model_natural.load_state_dict(torch.load(
        code_directory+f'AdversarialDecomposition/models/natural_{seed}.pt',
        map_location=dev()))
    model_natural.to(dev())
    model_natural.double()
    model_natural.eval()
    model_madry = model_loader.madry_diff()
    model_madry.load_state_dict(torch.load(
        code_directory+f'AdversarialDecomposition/models/robust_{seed}.pt',
        map_location=dev()))
    model_madry.to(dev())
    model_madry.double()
    model_madry.eval()
    return model_natural, data_natural, model_madry, data_madry


def load_cifar(code_directory):
    # load data
    #data_natural = np.load(code_directory+'AdversarialDecomposition/data/cifar_natural_diff.npy', allow_pickle=True).item()
    #data_madry = np.load(code_directory+'AdversarialDecomposition/data/cifar_robust_diff.npy', allow_pickle=True).item()
    data_natural = np.load(code_directory+'AdversarialDecomposition/data/cifar_natural_wrn.npy', allow_pickle=True).item()
    data_madry = np.load(code_directory+'AdversarialDecomposition/data/cifar_robust_wrn.npy', allow_pickle=True).item()
    # load models
    # natural
    ds = CIFAR(code_directory+'AdversarialDecomposition/data/cifar-10-batches-py')
    resume_path = code_directory+'AdversarialDecomposition/models/nat_diff_new.pt'
    model_natural = model_zoo.WideResNet(
        num_classes=10, depth=70, width=16,
        activation_fn=model_zoo.Swish,
        mean=model_zoo.CIFAR10_MEAN,
        std=model_zoo.CIFAR10_STD)

    # The model was trained without biases for the batch norm (thankfully those are initialized to zero) :/
    params = torch.load(resume_path)
    model_natural.load_state_dict(params, strict=False)
    model_natural.to(dev())
    model_natural.double()
    model_natural.eval()

    #classifier_model = ds.get_model('resnet50', False)
    #model_natural = model_loader.cifar_pretrained(classifier_model, ds)
    #resume_path = code_directory+'AdversarialDecomposition/models/nat_diff.pt'
    #checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))
    #state_dict_path = 'model'
    #if not ('model' in checkpoint):
    #    state_dict_path = 'state_dict'
    #sd = checkpoint[state_dict_path]
    #sd = {k[len('module.'):]: v for k, v in sd.items()}
    #model_natural.load_state_dict(sd)
    #model_natural.to(dev())
    #model_natural.double()
    #model_natural.eval()

    # madry
    ds = CIFAR(code_directory+'AdversarialDecomposition/data/cifar-10-batches-py')
    resume_path = code_directory+'AdversarialDecomposition/models/rob_diff_new.pt'
    model_madry = model_zoo.WideResNet(
        num_classes=10, depth=70, width=16,
        activation_fn=model_zoo.Swish,
        mean=model_zoo.CIFAR10_MEAN,
        std=model_zoo.CIFAR10_STD)

    # The model was trained without biases for the batch norm (thankfully those are initialized to zero) :/
    params = torch.load(resume_path)
    model_madry.load_state_dict(params, strict=False)
    model_madry.to(dev())
    model_madry.double()
    model_madry.eval()

    #classifier_model = ds.get_model('resnet50', False)
    #model_madry = model_loader.cifar_pretrained(classifier_model, ds)
    #resume_path = code_directory+'AdversarialDecomposition/models/rob_diff.pt'
    #checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))
    #state_dict_path = 'model'
    #if not ('model' in checkpoint):
    #    state_dict_path = 'state_dict'
    #sd = checkpoint[state_dict_path]
    #sd = {k[len('module.'):]:v for k,v in sd.items()}
    #model_madry.load_state_dict(sd)
    #model_madry.to(dev())
    #model_madry.double()
    #model_madry.eval()

    return model_natural, data_natural, model_madry, data_madry
