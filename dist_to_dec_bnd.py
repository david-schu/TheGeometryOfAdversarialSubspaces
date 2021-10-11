import numpy as np
from utils import load_model

from utils import get_dist_dec

import tqdm

dset='CIFAR'
is_natural = 1

#load model
if is_natural:
    model_path = './../models/cifar_models/nat_diff.pt'
    data_path = './data/cifar_natural_diff.npy'
else:
    model_path = './../models/cifar_models/rob_diff.pt'
    data_path = './data/cifar_robust_diff.npy'

model = load_model(resume_path=model_path, dataset=dset)

# load data
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
