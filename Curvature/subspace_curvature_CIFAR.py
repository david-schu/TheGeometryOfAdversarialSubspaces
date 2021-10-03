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

sys.path.insert(0, './../..')

import response_contour_analysis.utils.model_handling as model_utils
import response_contour_analysis.utils.principal_curvature as curve_utils

if __name__ == "__main__":
    hess_params = dict()
    hess_params['hessian_num_pts'] = 1.0e4
    hess_params['hessian_lr'] = 1e-4
    hess_params['hessian_random_walk'] = False
    hess_params['return_points'] = False
    hess_params['lr_decay'] = False#True
    hess_radius_mult = 0.7 # times the min adv perturbation length
    num_iters = 2 # for paired image boundary search
    num_steps_per_iter = 100 # for paired image boundary search
    buffer_portion = 0.25
    num_eps = 1000

    batch_size = 10

    num_images = 50
    num_advs = 10
    autodiff = True
    
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
    
    plot_colors = [tab_name_to_hex('tab:blue'), tab_name_to_hex('tab:red')]
    seed = 0
    # load data
    data_natural = np.load(f'/home/bethge/dpaiton/Work/AdversarialDecomposition/data/cifar_natural_diff.npy', allow_pickle=True).item()
    advs_nat = data_natural['advs']
    pert_lengths_nat = data_natural['pert_lengths']
    classes_nat = data_natural['adv_class']
    dirs_nat = data_natural['dirs']
    images_nat = data_natural['images']
    labels_nat = data_natural['labels']
    
    data_madry = np.load(f'/home/bethge/dpaiton/Work/AdversarialDecomposition/data/cifar_robust_diff.npy', allow_pickle=True).item()
    advs_madry = data_madry['advs']
    pert_lengths_madry = data_madry['pert_lengths']
    classes_madry = data_madry['adv_class']
    dirs_madry = data_madry['dirs']
    images_madry = data_madry['images']
    labels_madry = data_madry['labels']
    
    cifar_labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    sub_pert_lengths_nat = pert_lengths_nat[:, :num_advs]
    sub_pert_lengths_madry = pert_lengths_madry[:, :num_advs]
    mean_pert_lengths = np.mean([sub_pert_lengths_nat[np.isfinite(sub_pert_lengths_nat)].mean(),
        sub_pert_lengths_madry[np.isfinite(sub_pert_lengths_madry)].mean()])
    min_pert_lengths = np.min([sub_pert_lengths_nat[np.isfinite(sub_pert_lengths_nat)].min(),
        sub_pert_lengths_madry[np.isfinite(sub_pert_lengths_madry)].min()])
    hess_params['hessian_dist'] = min_pert_lengths * hess_radius_mult # radius around the target image
    
    # load models
    ds = CIFAR('/home/bethge/dpaiton/Work/AdversarialDecomposition/data/cifar-10-batches-py')
    classifier_model = ds.get_model('resnet50', False)
    model_natural = model_loader.cifar_pretrained(classifier_model, ds)
    
    resume_path = '/home/bethge/dpaiton/Work/AdversarialDecomposition/models/nat_diff.pt'
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
    classifier_model = ds.get_model('resnet50', False)
    model_madry = model_loader.cifar_pretrained(classifier_model, ds)
    resume_path = '/home/bethge/dpaiton/Work/AdversarialDecomposition/models/rob_diff.pt'
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
    
    def torchify(img):
        output = torch.from_numpy(img).type(torch.DoubleTensor).to(dev()) # always autodiff
        output.requires_grad = True
        return output
    
    
    def get_paired_boundary_image(model, origin, alt_image, num_steps_per_iter, num_iters):
        input_shape = (1,)+origin.shape
        def find_pert(image_line):
            correct_lbl = torch.argmax(model(torchify(image_line[0, ...].reshape(input_shape))))
            pert_lbl = correct_lbl.clone()
            step_idx = 1 # already know the first one
            while pert_lbl == correct_lbl:
                pert_image = image_line[step_idx, ...]
                pert_lbl = torch.argmax(model(torchify(pert_image.reshape(input_shape))))
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
    
    
    def generate_paired_dict(data_dict, model, num_images, num_advs):
        image_shape = data_dict['images'].shape[1:]
        num_pixels = int(np.prod(image_shape))
        images = np.zeros((num_images,) + image_shape)
        labels = np.zeros((num_images), dtype=np.int)
        dirs = np.zeros((num_images, num_advs, 1, num_pixels))
        advs = np.zeros((num_images, num_advs, 1, num_pixels))
        pert_lengths = np.zeros((num_images, num_advs))
        adv_class = np.zeros((num_images, num_advs))
        #model_predictions = torch.argmax(model(torchify(data_dict['images'])), dim=1).detach().cpu().numpy()
        batch_size = 10
        image_splits = torch.split(torchify(data_dict['images']), batch_size)
        model_predictions = []
        for batch_idx, batch in enumerate(image_splits):
            model_predictions.append(torch.argmax(model(batch), dim=1).detach().cpu().numpy())
        model_predictions = np.stack(model_predictions, axis=0).reshape((len(model_predictions)*batch_size,) + model_predictions[0].shape[1:])
        valid_indices = []
        for image_idx in range(data_dict['images'].shape[0]):
            if model_predictions[image_idx] == data_dict['labels'][image_idx]:
                valid_indices.append(image_idx)
        origin_indices = np.random.choice(valid_indices, size=num_images, replace=False)
        for image_idx, origin_idx in enumerate(origin_indices):
            images[image_idx, ...] = data_dict['images'][origin_idx, ...]
            labels[image_idx] = data_dict['labels'][origin_idx]
            shuffled_valid_indices = np.random.choice(valid_indices, size=len(valid_indices), replace=False)
            alt_indices = [idx for idx, alt_class in zip(shuffled_valid_indices, data_dict['labels'][shuffled_valid_indices]) if alt_class != labels[image_idx]]
            for dir_idx, alt_idx in enumerate(alt_indices[:num_advs]):
                alt_image = data_dict['images'][alt_idx, ...]
                boundary_image, boundary_dir, pert_length = get_paired_boundary_image(
                    model, images[image_idx, ...], alt_image, num_steps_per_iter, num_iters)
                dirs[image_idx, dir_idx, ...] = boundary_dir.reshape(1, -1)
                advs[image_idx, dir_idx, ...] = boundary_image.reshape(1, -1)
                adv_class[image_idx,  dir_idx] = torch.argmax(model(torchify(boundary_image[None, ...]))).item()
                pert_lengths[image_idx, dir_idx] =  pert_length
        output_dict = {}
        output_dict['images'] = images
        output_dict['labels'] = labels
        output_dict['dirs'] = dirs
        output_dict['advs'] = advs
        output_dict['adv_class'] = adv_class
        output_dict['pert_lengths'] = pert_lengths
        return output_dict
    
    
    def paired_activation(model, image, neuron1, neuron2):
        if not image.requires_grad:
            image.requires_grad = True
        model.zero_grad()
        activation1 = model_utils.unit_activation(model, image, neuron1, compute_grad=True)
        activation2 = model_utils.unit_activation(model, image, neuron2, compute_grad=True)
        activation_difference = activation1 - activation2
        return activation_difference
    
    
    def paired_activation_and_gradient(model, image, neuron1, neuron2):
        activation_difference = paired_activation(model, image, neuron1, neuron2)
        grad = torch.autograd.grad(activation_difference, image)[0]
        return activation_difference, grad
    
    
    def get_curvature(condition_zip, num_images, num_advs, num_eps, batch_size, buffer_portion, autodiff=False):
        """
        A note on the gradient of the difference in activations:
        The gradient points in the direction of the origin from the boundary image.
        Therefore, for large enough eps, origin - eps * grad/|grad| will reach the boundary; and boundary + eps * grad/|grad| will reach the origin 
        """
        models, model_data = zip(*condition_zip)
        num_models = len(models)
        image_shape = model_data[0]['images'][0, ...][None, ...].shape
        image_size = np.prod(image_shape)
        num_dims = image_size - 1 #removes normal direction
        shape_operators = np.zeros((num_models, num_images, num_advs, num_dims, num_dims))
        principal_curvatures = np.zeros((num_models, num_images, num_advs, num_dims))
        principal_directions = np.zeros((num_models, num_images, num_advs, image_size, num_dims))
        origin_indices = np.zeros((num_models, num_images), dtype=np.int)
        for model_idx, (model_, data_)  in enumerate(zip(models, model_data)):
            model_predictions = torch.argmax(model_(torchify(data_['images'])), dim=1).detach().cpu().numpy()
            valid_indices = [] # Need to ensure that all images are correctly labeled & have valid adversarial examples
            for image_idx in range(data_['images'].shape[0]):
                if model_predictions[image_idx] == data_['labels'][image_idx]: # correctly labeled
                    if np.all(np.isfinite(data_['pert_lengths'][image_idx, :num_advs])): # enough adversaries found
                        valid_indices.append(image_idx)
            origin_indices[model_idx, :] = np.random.choice(valid_indices, size=num_images, replace=False)
            pbar = tqdm(total=num_advs*num_images, leave=False)
            for image_idx, origin_idx in enumerate(list(origin_indices[model_idx, :])):
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
                    if autodiff:
                        def func(x):
                            acts_diff = paired_activation(model_, x, clean_lbl, adv_lbl)
                            return acts_diff
                        hessian = torch.autograd.functional.hessian(func, torchify(boundary_image[None,...]))
                        hessian = hessian.reshape((int(boundary_image.size), int(boundary_image.size)))
                    else:
                        def func(x):
                            acts_diff, grad = paired_activation_and_gradient(model_, x, clean_lbl, adv_lbl)
                            return acts_diff, grad
                        hessian = curve_utils.sr1_hessian(
                            func, torchify(boundary_image[None, ...]),
                            distance=hess_params['hessian_dist'],
                            n_points=hess_params['hessian_num_pts'],
                            random_walk=hess_params['hessian_random_walk'],
                            learning_rate=hess_params['hessian_lr'],
                            return_points=False,
                            progress=False)
                    activation, gradient = paired_activation_and_gradient(model_, torchify(boundary_image[None, ...]), clean_lbl, adv_lbl)
                    gradient = gradient.reshape(-1)
                    curvature = curve_utils.local_response_curvature_isoresponse_surface(gradient, hessian)
                    shape_operators[model_idx, image_idx, adv_idx, ...] = curvature[0].detach().cpu().numpy()
                    principal_curvatures[model_idx, image_idx, adv_idx, :] = curvature[1].detach().cpu().numpy()
                    principal_directions[model_idx, image_idx, adv_idx, ...] = curvature[2].detach().cpu().numpy()
                    pbar.update(1)
        pbar.close()
        return shape_operators, principal_curvatures, principal_directions, origin_indices
    
    
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
    
    
    if autodiff:
        filename = '/home/bethge/dpaiton/Work/AdversarialDecomposition/data/cifar_curvatures_and_directions_autodiff.npz'
    else:
        filename = '/home/bethge/dpaiton/Work/AdversarialDecompsition/data/cifar_curvatures_and_directions_sr1.npz'
    
    data_paired_natural = generate_paired_dict(data_natural, model_natural, num_images, num_advs)
    data_paired_madry = generate_paired_dict(data_madry, model_madry, num_images, num_advs)
    paired_condition_zip = zip([model_natural, model_madry], [data_paired_natural, data_paired_madry])
    paired_shape_operators, paired_principal_curvatures, paired_principal_directions, paired_origin_indices = get_curvature(
        paired_condition_zip, num_images, num_advs, num_eps, batch_size, buffer_portion, autodiff)
    paired_mean_curvatures = np.mean(paired_principal_curvatures, axis=-1)
    adv_condition_zip = zip([model_natural, model_madry], [data_natural, data_madry])
    adv_shape_operators, adv_principal_curvatures, adv_principal_directions, adv_origin_indices = get_curvature(
        adv_condition_zip, num_images, num_advs, num_eps, batch_size, buffer_portion, autodiff)
    adv_mean_curvatures = np.mean(adv_principal_curvatures, axis=-1)
    save_dict = {
        'data_paired_natural':data_paired_natural,
        'data_paired_madry':data_paired_madry,
        'paired_shape_operators': paired_shape_operators,
        'paired_principal_curvatures': paired_principal_curvatures,
        'paired_principal_directions': paired_principal_directions,
        'paired_mean_curvatures': paired_mean_curvatures,
        'paired_origin_indices': paired_origin_indices,
        'adv_shape_operators': adv_shape_operators,
        'adv_principal_curvatures': adv_principal_curvatures,
        'adv_principal_directions': adv_principal_directions,
        'adv_mean_curvatures': adv_mean_curvatures,
        'adv_origin_indices': adv_origin_indices
    }
    np.savez(filename, data=save_dict)
    
    filename = '/home/bethge/dpaiton/Work/AdversarialDecomposition/data/cifar_subspace_curvatures_autodiff.npz'
    
    dtype = torch.double
    subspace_size = num_advs-1
    rand_pcs = np.zeros((num_models, num_images, subspace_size))
    adv_pcs = np.zeros((num_models, num_images, subspace_size))
    for model_idx, (model_, data_) in enumerate(zip((model_natural, model_madry), (data_natural, data_madry))):
        pbar = tqdm(total=num_advs*num_images, leave=False)
        for image_idx, origin_idx in enumerate(adv_origin_indices[model_idx, :]):
            for adv_idx in range(num_advs):
                boundary_image, boundary_dir, pert_length = get_paired_boundary_image(
                    model=model_,
                    origin=data_['images'][origin_idx, ...],
                    alt_image=data_['advs'][origin_idx, adv_idx, ...],
                    num_steps_per_iter=num_steps_per_iter,
                    num_iters=num_iters)
                clean_lbl = int(data_['labels'][origin_idx])
                adv_lbl = int(data_['adv_class'][origin_idx, adv_idx])
                activation, gradient = paired_activation_and_gradient(model_, torchify(boundary_image[None, ...]), clean_lbl, adv_lbl)
                gradient = gradient.reshape(-1).type(dtype)
                def func(x):
                    acts_diff = paired_activation(model_, x, clean_lbl, adv_lbl)
                    return acts_diff
                hessian = torch.autograd.functional.hessian(func, torchify(boundary_image[None,...]))
                hessian = hessian.reshape((int(boundary_image.size), int(boundary_image.size))).type(dtype)
                dirs = [(gradient / torch.linalg.norm(gradient)).detach().cpu().numpy()]
                n_pixels = gradient.numel()
                n_iterations = 3
                random_basis = torch.from_numpy(make_orth_basis(dirs, n_pixels, n_iterations)[:subspace_size, :]).type(dtype)
                curvature = curve_utils.local_response_curvature_isoresponse_surface(gradient, hessian, projection_subspace_of_interest=random_basis)
                rand_subspace_curvatures = curvature[1].detach().cpu().numpy()
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
                rand_pcs[model_idx, image_idx, :] = rand_subspace_curvatures
                adv_pcs[model_idx, image_idx, :] = adv_subspace_curvatures
            pbar.update(1)
        pbar.close()
    save_dict = {
        'random_principal_curvatures':rand_pcs,
        'adversarial_principal_curvatures':adv_pcs,
    }
    np.savez(filename, data=save_dict)
