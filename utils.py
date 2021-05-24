import numpy as np
import torch
import torchvision.datasets as datasets

def orth_check(adv_dirs):
    orth = np.eye(len(adv_dirs))
    for i in range(len(adv_dirs)):
        for j in range(i + 1, len(adv_dirs)):
            orth[i, j] = np.dot(adv_dirs[i].T, adv_dirs[j])
            orth[j, i] = orth[i, j]
    return orth


def classification(img, model):
    if not torch.is_tensor(img):
        img = torch.tensor(img, device=dev())
    pred = model(img).cpu().detach().numpy()
    img_class = np.argmax(pred, axis=-1)
    return img_class


def dirs_to_attack_format(dirs):
    max_dim = max(len(elem) for elem in dirs)
    attack_dirs = torch.zeros([len(dirs), max_dim, dirs[0].shape[-1]], device=dev())
    for i, d in enumerate(dirs):
        attack_dirs[i, :len(d)] = d
    return attack_dirs


def load_data(n, bounds=(0., 1.), random=True):
    mnist = datasets.MNIST(root='../data', download=True)
    indices = np.arange(len(mnist.data))
    if random:
        np.random.shuffle(indices)
    images = mnist.data[indices[:n]]
    images = images / 255 * (bounds[1] - bounds[0]) + bounds[0]
    images = images.unsqueeze(1)
    labels = mnist.targets[indices[:n]]

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

