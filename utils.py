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


def dirs_to_attack_format(dirs):
    max_dim = max(len(elem) for elem in dirs)
    attack_dirs = torch.zeros([len(dirs), max_dim, dirs[0].shape[-1]], device=dev(), requires_grad=False)
    for i, d in enumerate(dirs):
        attack_dirs[i, :len(d)] = d
    return attack_dirs


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

def make_orth_basis(dirs=[]):
    n_iterations = 3
    n_pixel = 784  # dirs.shape[-1]
    basis = np.random.uniform(-1, 1, (n_pixel - len(dirs), n_pixel))
    basis = basis / np.linalg.norm(basis, axis=-1, keepdims=True)
    if len(dirs) > 0:
        basis_with_dirs = np.concatenate((dirs, basis), axis=0)
    else:
        basis_with_dirs = basis

    for it in range(n_iterations):
        for i, v in enumerate(basis):
            v_orth = v - ((basis_with_dirs[:len(dirs) + i] * v.reshape((1, -1))).sum(-1, keepdims=True) *
                          basis_with_dirs[:len(dirs) + i]).sum(0)
            u_orth = v_orth / np.linalg.norm(v_orth)
            basis_with_dirs[len(dirs) + i] = u_orth
            basis[i] = u_orth

    return basis