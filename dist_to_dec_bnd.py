import numpy as np
from models import model as md
import torch
from robustness1.datasets import CIFAR
import dill
import sys

from utils import dev, get_dist_dec

import tqdm


if __name__ == "__main__":
    is_natural = 1#int(sys.argv[1])

    if is_natural:
        resume_path = './models/cifar_models/nat_diff.pt'
    else:
        resume_path = './models/cifar_models/rob_diff.pt'

    # Load model
    ds = CIFAR('./data/cifar-10-batches-py')
    classifier_model = ds.get_model('resnet50', False)
    model = md.cifar_pretrained(classifier_model, ds)

    checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))

    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'
    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(dev())
    model.double()
    model.eval()

    # load data
    if is_natural:
        data_path = './data/cifar_natural_diff.npy'
    else:
        data_path = './data/cifar_robust_diff.npy'
    data = np.load(data_path, allow_pickle=True).item()
    pert_lengths = data['pert_lengths']
    dirs = data['dirs']
    images = data['images']
    labels = data['labels']

    n_samples = 100
    n_dims = 50

    #natural
    min_dists = pert_lengths[:, 0]

    dists = np.zeros((len(images), n_dims, n_samples))
    angles= np.zeros((len(images), n_dims, n_samples))
    largest_vecs = np.zeros((len(images), n_dims, dirs.shape[-1]))
    for i, img in enumerate(tqdm.tqdm(images)):
        for n in np.arange(0, n_dims):
            dists[i, n], angles[i, n], largest_vecs[i, n] = get_dist_dec(img, labels[i], dirs[i, :n + 1], model,
                                                                         min_dist=0.5 * min_dists[i],
                                                                         n_samples=n_samples)

        data = {
            'dists': dists,
            'angles': angles,
            'largest_vecs': largest_vecs
        }
        if is_natural:
            save_path = './data/dists_to_bnd_natural.npy'
        else:
            save_path = './data/dists_to_bnd_robust.npy'
        np.save(save_path, data)
