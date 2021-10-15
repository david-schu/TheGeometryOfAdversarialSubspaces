import numpy as np
import torch
import torchvision.datasets as datasets
from models.mnist_models import model as model_loader
from robustness.datasets import CIFAR
from models.cifar_models import model_utils


def classification(img, model, is_adv=True, orig_label=None):
    """
    Classify a image with a given model and optionally make sure the  image has an adversarial class
    Parameters:
        img [torch tensor] the image to be classified
        model [PyTorch model] the classification model
        is_adv [Bool] wheter missclassification is required
        orig_label [int] the original class label if the input is an adversarial - required when is_adv=True
    Outputs:
        dataset_index [int] model output class
    """
    if not torch.is_tensor(img):
        img = torch.tensor(img, device=dev())
    pred = model(img).cpu().detach().numpy()[0]
    img_class = np.argmax(pred)
    if is_adv:
        assert img_class != orig_label
    return img_class


def load_raw_data(n, bounds=(0., 1.), random=True, d_set='MNIST', train=True):
    """
    Load raw MNIST or CIFAR-10 data
    Parameters:
        n [int] the number of samples to be loaded
        bounds [tuple] the image bounds of the model
        random [Bool] specifies if dataset should be shuffled before loading
        d_set [string] the required datasat - either "CIFAR" or "MNIST"
        train [Bool] if True, trainig data is laoded, else, test data
    Outputs:
        images [torch tensor] a set of torch dataset samples
        labels [torch tesnor] a set of according labels to the samples
    """
    if d_set == 'MNIST':
        dset = datasets.MNIST(root='../data', train=train, download=True)
    elif d_set == 'CIFAR':
        dset = datasets.CIFAR10(root='../data', train=train, download=True)
    indices = np.arange(len(dset.data))
    if random:
        np.random.shuffle(indices)
    images = dset.data[indices[:n]]
    images = images / 255 * (bounds[1] - bounds[0]) + bounds[0]

    if d_set == 'MNIST':
        images = images.unsqueeze(1)
        labels = dset.targets[indices[:n]]
    elif d_set == 'CIFAR':
        images = images.transpose(0, 3, 1, 2)
        labels = np.array(dset.targets)[indices[:n]]
    images = torch.as_tensor(images)
    labels = torch.as_tensor(labels)
    return images.double(), labels


def load_stable_data(d_set='MNIST'):
    """
    Load balanced and correctly classified MNIST or CIFAR-10 data
    Parameters:
        d_set [string] the required datasat - either "CIFAR" or "MNIST"
    Outputs:
        data [dict] a dictionary that includes data which is balanced and correctly classified for all relevant models
    """
    if d_set == 'MNIST':
        data = np.load('./data/MNIST/stable_data.npy', allow_pickle=True).item()
    elif d_set == 'CIFAR':
        data = np.load('./data/CIFAR/stable_data.npy', allow_pickle=True).item()
    else:
        raise ValueError('Invalid Dataset')
    return data


def load_model(resume_path, dataset):
    """
    Load a PyTorch model
    Parameters:
        resume_path [string] path to the .pt file of the model
        d_set [string] the required datasat - either "CIFAR" or "MNIST"
    Outputs:
        model [PyTorch model] a classification model
    """
    if dataset == 'MNIST':
        model = model_loader.Madry()
        model.load_state_dict(
            torch.load(resume_path, map_location=torch.device(dev())))
        model.to(dev())
        model.double()
        model.eval()

    elif dataset == 'CIAFR':
        ds = CIFAR('./data')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
        model.to(dev())
        model.double()
        model.eval()
    else:
        raise ValueError('Invalid Dataset')
    return model


def dev():
    """
    Returns the current device
    """
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


def map_to(x, tmin, tmax, rmin=0, rmax=1):
    """
    Map an array to new range
    Parameters:
        x [array] array to be mapped
        tmin [double] the target lower bound
        tmax [double] the target upper bound
        rmin [double] the original lower bound
        rmax [double] the original upper bound
    Outputs:
        x_t [array] the mapped array
    """
    if tmin == tmax:
        x_t = x*0+tmin
    else:
        x_t = (x-rmin)*(tmax-tmin)/(rmax-rmin)+tmin
    return x_t


def make_orth_basis(dirs=[], n_pixels=784, n_iterations=3):
    """
    Make an orthogonal basis covering the whole input space given to some optional vectors - given vectors must be
    orthogonal already
    Parameters:
        dirs [array] list of orthogonal directions
        n_pixels [int] size of the input space
        n_iterations [int] iterations of gram-schmidt for precision
    Outputs:
        basis [array] an orthogonal basis given some vectors
    """
    basis = np.random.uniform(-1, 1, (n_pixels - len(dirs), n_pixels))
    basis = basis / np.linalg.norm(basis, axis=-1, keepdims=True)
    if len(dirs) > 0:
        basis_with_dirs = np.concatenate((dirs, basis), axis=0)
    else:
        basis_with_dirs = basis.copy()

    for it in range(n_iterations):
        for i, v in enumerate(basis):
            v_orth = v - ((basis_with_dirs[:len(dirs) + i] * v.reshape((1, -1))).sum(-1, keepdims=True) *
                          basis_with_dirs[:len(dirs) + i]).sum(0)
            u_orth = v_orth / np.linalg.norm(v_orth)
            basis_with_dirs[len(dirs) + i] = u_orth
            basis[i] = u_orth
    return basis


def get_dist_dec(orig, label, dirs, model, min_dist=.1, n_samples=1000):
    """
    Get the distance to the decision boundary from the original image for randomly sampled vectors of the positive
    space spanned by given vectors. To find the distance a n_step binary search is performed. Also returns the smallest
    angles of the random directions to the given vectors and the vector with the largest distance to the decision
    boundary.
    Parameters:
        orig [array] original image sample
        label [int] original label of sample
        dirs [double] adversarial directions that span the subspace in which distance is evaluated
        model [PyTorch model] the classification model
        min_dist [double] the distance at which to start the search
        n_samples [int] the number of randomly sampled directions in the subspace
    Outputs:
        dists [array] the distances of to the decision boundary of the sampled vectors
        angles [array] the smallest angles of the sampled vectors to the given input directions
        largest_vec [array] the vector with the largest distance to the decision boundary
    """
    shape = orig.shape
    n_steps = 20
    n_dirs = len(dirs)
    dirs = dirs.reshape((n_dirs, -1))

    upper = np.full((n_samples, 1), np.inf)
    lower = np.ones((n_samples, 1)) * min_dist

    scales = np.ones((n_samples, 1)) * min_dist

    coeffs = abs(np.random.normal(size=[n_samples, n_dirs]))
    sample_dirs = (coeffs @ dirs)
    sample_dirs = sample_dirs / np.linalg.norm(sample_dirs, axis=-1, keepdims=True)

    dists = np.full(n_samples, np.nan)
    found_advs = np.full(n_samples, False)
    for i in range(n_steps):
        input_dirs = scales * sample_dirs
        input_ = input_dirs + orig.flatten()[None]
        input = torch.split(torch.tensor(input_.reshape((-1,) + shape), device=dev()), 100)

        preds = np.empty(0)
        for batch in input:
            preds = np.concatenate((preds, model(batch).argmax(-1).cpu().numpy()), axis=0)

        is_adv = np.invert(preds == label)
        found_advs[is_adv] = True
        dists[is_adv] = scales[is_adv, 0]

        upper[is_adv] = scales[is_adv]
        lower[~is_adv] = scales[~is_adv]
        scales[found_advs] = (upper[found_advs] + lower[found_advs]) / 2
        scales[~found_advs] = lower[~found_advs] * 2

        in_bounds = np.logical_and(input_.max(-1) <= 1, input_.min(-1) >= 0)
        dists[~in_bounds] = np.nan
    if np.all(np.isnan(dists)):
        largest_vec = np.zeros(dirs.shape[-1])
    else:
        largest_vec = sample_dirs[np.nanargmax(dists)]
    angles = np.arccos((sample_dirs@dirs.T).clip(-1,1)).min(-1)
    angles[np.isnan(dists)] = np.nan

    return dists, angles, largest_vec