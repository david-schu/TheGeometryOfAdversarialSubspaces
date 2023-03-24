import numpy as np
import torch
import sys

from utils import load_model
from utils import get_dist_dec

import tqdm

if __name__ == "__main__":
    batch_n = int(sys.argv[1])
    is_natural = int(sys.argv[2])
    dset = 'CIFAR'
    batchsize = 100
    with torch.no_grad():
        #load model
        if is_natural:
            model_path = './models/cifar_models/nat_resNet50.pt'
            data_path = './data/CIFAR/cifar_natural_resNet50.npy'
            save_path = './data/CIFAR/dists_to_bnd_natural_resNet50_' + str(batch_n) + '.npy'
        else:
            model_path = './models/cifar_models/rob_resNet50.pt'
            data_path = './data/CIFAR/cifar_robust_resNet50.npy'
            save_path = './data/CIFAR/dists_to_bnd_robust_resNet50_' + str(batch_n) + '.npy'

        model = load_model(resume_path=model_path, dataset=dset)

        # load data
        data = np.load(data_path, allow_pickle=True).item()
        pert_lengths = data['pert_lengths'][(batch_n*batchsize):(batch_n*batchsize+batchsize)]
        dirs = data['dirs'][(batch_n*batchsize):(batch_n*batchsize+batchsize)]
        images = data['images'][(batch_n*batchsize):(batch_n*batchsize+batchsize)]
        labels = data['labels'][(batch_n*batchsize):(batch_n*batchsize+batchsize)]

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
            np.save(save_path, data)
