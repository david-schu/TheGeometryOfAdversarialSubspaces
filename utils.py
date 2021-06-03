import numpy as np
import torch
import torchvision.datasets as datasets

def orth_check(adv_dirs):
    orth = np.dot(adv_dirs,adv_dirs.T)
    return orth, np.allclose(orth, np.identity(orth.shape[0]))


def classification(img, model):
    if not torch.is_tensor(img):
        img = torch.tensor(img, device=dev())
    pred = model(img).cpu().detach().numpy()
    img_class = np.argmax(pred, axis=-1)
    return img_class


def dirs_to_attack_format(dirs):
    max_dim = max(len(elem) for elem in dirs)
    attack_dirs = torch.zeros([len(dirs), max_dim, dirs[0].shape[-1]], device=dev(), requires_grad=False)
    for i, d in enumerate(dirs):
        attack_dirs[i, :len(d)] = d
    return attack_dirs


def load_data(n, bounds=(0., 1.), random=True, d_set='MNIST'):
    if d_set=='MNIST':
        dset = datasets.MNIST(root='../data', download=True)
    elif d_set == 'CIFAR':
        dset = datasets.CIFAR10(root='../data', download=True)
    indices = np.arange(len(dset.data))
    if random:
        np.random.shuffle(indices)
    images = dset.data[indices[:n]]
    images = images / 255 * (bounds[1] - bounds[0]) + bounds[0]

    if d_set=='MNIST':
        images = images.unsqueeze(1)
        labels = dset.targets[indices[:n]]
    elif d_set == 'CIFAR':
        images = images.transpose(0, 3, 1, 2)
        labels = np.array(dset.targets)[indices[:n]]
    images = torch.as_tensor(images, device=dev())
    labels = torch.as_tensor(labels, device=dev())
    return images, labels


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

