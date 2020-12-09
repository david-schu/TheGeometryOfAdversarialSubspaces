import numpy as np
import torch
import torchvision.datasets as datasets
from abs_models import utils as u


def orth_check(adv_dirs):
    orth = np.eye(len(adv_dirs))
    for i in range(len(adv_dirs)):
        for j in range(i + 1, len(adv_dirs)):
            orth[i, j] = np.dot(adv_dirs[i].T, adv_dirs[j])
            orth[j, i] = orth[i, j]
    return orth


def classification(img, model):
    if not torch.is_tensor(img):
        img = torch.tensor(img)
    pred = u.t2n(model(img))
    img_class = np.argmax(pred, axis=-1)
    return img_class


def dirs_to_attack_format(dirs):
    max_dim = max(len(elem) for elem in dirs)
    attack_dirs = np.zeros([len(dirs), max_dim, dirs[0].shape[-1]])
    for i, dir in enumerate(dirs):
        attack_dirs[i, :len(dir)] = dir
    return torch.tensor(attack_dirs,device=u.dev())


def load_data(n, bounds=(0., 1.)):
    mnist = datasets.MNIST(root='./data', download=True)
    images = mnist.data[:n]
    images = images / 255 * (bounds[1] - bounds[0]) + bounds[0]
    images = images.unsqueeze(1)
    labels = mnist.targets[:n]

    images = torch.as_tensor(images,device=u.dev())
    labels = torch.as_tensor(labels, device=u.dev())
    return images, labels

def load_batched_data(n,batchsize,bounds=(0.,1.)):
    images,labels = load_data(n=n,bounds=bounds)
    batched_images = torch.split(images, batchsize, dim=0)
    batched_labels = torch.split(labels, batchsize, dim=0)
    return batched_images,batched_labels
