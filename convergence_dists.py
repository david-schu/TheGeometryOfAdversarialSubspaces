import numpy as np
from models import model as md
import torch
from robustness1.datasets import CIFAR
import dill

from utils import dev, get_dist_dec

import tqdm


if __name__ == "__main__":

    resume_path = './models/cifar_models/nat_diff.pt'

    # Load model
    ds = CIFAR('./data/cifar-10-batches-py')
    classifier_model = ds.get_model('resnet50', False)
    model = md.CifarPretrained(classifier_model, ds)

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
    data_path = './data/cifar_natural_diff.npy'
    data = np.load(data_path, allow_pickle=True).item()
    pert_lengths = data['pert_lengths']
    dirs = data['dirs']
    images = data['images']
    labels = data['labels']

    n_samples = np.linspace(5, 100, 5)
    n_dims = 50

    #natural
    min_dists = pert_lengths[:, 0]
    all_dists = []

    for n in n_samples:
        dists = np.zeros(len(images), n)
        for i, img in enumerate(tqdm.tqdm(images)):
            dists[i], _, _ = get_dist_dec(img, labels[i], dirs[i, :n_dims], model, min_dist=0.5 * min_dists[i],
                                          n_samples=n)
            all_dists.append(dists)

        data = {
            'dists': all_dists,
        }

        save_path = './data/smaple_convergence.npy'
        np.save(save_path, data)