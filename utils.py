import numpy as np
import torch
import torchvision.datasets as datasets


def orth_check(adv_dirs):
    adv_dirs = np.array(adv_dirs).reshape((adv_dirs.shape[0],-1))
    orth = np.dot(adv_dirs,adv_dirs.T)
    return orth, np.allclose(orth, np.identity(orth.shape[0]),atol=1e-2)


def classification(img, label, model):
    if not torch.is_tensor(img):
        img = torch.tensor(img, device=dev())
    pred = model(img).cpu().detach().numpy()[0]
    sorted = np.sort(pred)
    img_class = np.argmax(pred)
    is_adv = label != img_class
    if not is_adv and int(sorted[-2] * 10000) == int(sorted[-1] * 10000):
        img_class = pred.argsort()[-2]
    return img_class


def load_data(n, bounds=(0., 1.), random=True, d_set='MNIST', train=True):
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


def dev():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


def map_to(x, tmin, tmax, rmin=0, rmax=1):
    if tmin == tmax:
        x_t = x*0+tmin
    else:
        x_t = (x-rmin)*(tmax-tmin)/(rmax-rmin)+tmin
    return x_t


def make_orth_basis(dirs=[], n_pixels=784, n_iterations=3):
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


def get_dist_dec(orig, label, dirs, model, min_dist=.1, n_samples=1000, return_angles=False):
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

    for i in range(n_steps):
        input_dirs = scales * sample_dirs
        input_ = (input_dirs + orig.flatten()[None])
        input = torch.split(torch.tensor(input_.reshape((-1,) + shape), device=dev()), 100)

        preds = np.empty(0)
        for batch in input:
            preds = np.concatenate((preds, model(batch).argmax(-1).cpu().numpy()), axis=0)

        is_adv = np.invert(preds == label)
        dists[is_adv] = scales[is_adv, 0]

        upper[is_adv] = scales[is_adv]
        lower[~is_adv] = scales[~is_adv]
        scales[is_adv] = (upper[is_adv] + lower[is_adv]) / 2
        scales[~is_adv] = lower[~is_adv] * 2

        in_bounds = np.logical_or(input_.max(-1) <= 1, input_.min(-1) >= 0)
        dists[~in_bounds] = np.nan

    if return_angles:
        angles = np.arccos((sample_dirs@dirs.T).clip(-1,1)).min(-1)
        angles[np.isnan(dists)] = np.nan
        return dists, angles
    return dists