import numpy as np
from models import model as md
import torch
import sys

from utils import dev, get_dist_dec

import tqdm


if __name__ == "__main__":
    is_natural = int(sys.argv[1])

    if is_natural:
        resume_path = './models/mnist_models/natural_0.pt'
    else:
        resume_path = './models/mnist_models/robust_0.pt'

    # load models
    model = md.madry_diff()
    model.load_state_dict(torch.load(resume_path, map_location=torch.device(dev())))
    model.to(dev())
    model.eval()

    # load data
    if is_natural:
        data_path = './data/MNIST_runs/natural_0.npy'
    else:
        data_path = './data/MNIST_runs/robust_0.npy'
    data = np.load(data_path, allow_pickle=True).item()
    pert_lengths = data['pert_lengths']
    dirs = data['dirs']
    images = data['images']
    labels = data['labels']

    n_samples = 1000
    n_dims = 8

    images = images[np.invert(np.isnan(pert_lengths)).sum(-1) >= n_dims]
    labels = labels[np.invert(np.isnan(pert_lengths)).sum(-1) >= n_dims]
    dirs = dirs[np.invert(np.isnan(pert_lengths)).sum(-1) >= n_dims]
    pert_lengths = pert_lengths[np.invert(np.isnan(pert_lengths)).sum(-1) >= n_dims]

    #natural
    min_dists = pert_lengths[:, 0]

    dists = np.zeros((len(images), n_dims, n_samples))
    angles= np.zeros((len(images), n_dims, n_samples))
    largest_vecs = np.zeros((len(images), n_dims, dirs.shape[-1]))

    for i, img in enumerate(tqdm.tqdm(images)):
        for n in np.arange(0, n_dims):
            dists[i, n], angles[i, n], largest_vecs[i, n] = get_dist_dec(img, labels[i], dirs[i, :n+1], model,
                                                min_dist=0.5 * min_dists[i], n_samples=n_samples)

        data = {
            'dists': dists,
            'angles': angles,
            'largest_vecs': largest_vecs
        }
        if is_natural:
            save_path = './data/MNIST_runs/dists_to_bnd_natural.npy'
        else:
            save_path = './data/MNIST_runs/dists_to_bnd_robust.npy'
        np.save(save_path, data)
