import sys
sys.path.insert(0, './..')

import torch
import numpy as np

from models import model
from utils import dev

def get_dist_dec(orig, label, dirs, model, n_samples=1000):
    n_scales = 100
    input_shape = (-1, 1, 28, 28)
    n_dirs = len(dirs)
    dirs = dirs.reshape((n_dirs, -1))
    scales = np.linspace(0, 10, n_scales)
    coeffs = abs(np.random.normal(size=[n_samples, n_dirs]))
    samples = (coeffs @ dirs)
    samples = samples/np.linalg.norm(samples, axis=-1, keepdims=True)
    dists = np.zeros(n_samples)
    for i, sample in enumerate(samples):
        input_dirs = np.outer(scales, sample)
        input = (input_dirs + orig.reshape((-1, 784)))
        pred = model(torch.tensor(input.reshape(input_shape), device=dev())).detach().cpu().numpy()
        pred_classes = np.argmax(pred, axis=-1)
        if np.all(pred_classes == label):
            dists[i] = np.nan
        else:
            idx = np.min(np.argwhere(pred_classes != label))
            new_scales = np.linspace(scales[idx-1], scales[idx], n_scales)
            input_dirs = np.outer(new_scales,sample).reshape(input_shape)
            input = (input_dirs + orig.reshape((-1,1,28,28)))
            pred = model(torch.tensor(
                input, device=dev())).detach().cpu().numpy()
            pred_classes = np.argmax(pred, axis=-1)
            idx = np.min(np.argwhere(pred_classes != label))
            if input[idx].max()>1 or input[idx].min()<0:
                dists[i] = np.nan
            else:
                dists[i] = np.linalg.norm(input_dirs[idx])
    return dists

n_dim = 8
n_samples = 100
seed = 0

model_natural = model.madry_diff()
model_natural.load_state_dict(torch.load(f'../models/natural_{seed}.pt', map_location=torch.device(dev())))
model_natural.to(dev())
model_natural.eval()

model_robust = model.madry_diff()
model_robust.load_state_dict(torch.load(f'../models/robust_{seed}.pt', map_location=torch.device(dev())))
model_robust.to(dev())
model_robust.eval()

# load data
data = np.load(f'../data/natural_{seed}.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']
pert_lengths = data['pert_lengths']

data = np.load(f'../data/robust_{seed}.npy', allow_pickle=True).item()
advs_madry = data['advs']
pert_lengths_madry = data['pert_lengths']
classes_madry = data['adv_class']
dirs_madry = data['dirs']
images_madry = data['images']
labels_madry = data['labels']

images_ = images[np.invert(np.isnan(pert_lengths)).sum(-1) > n_dim]
labels_ = labels[np.invert(np.isnan(pert_lengths)).sum(-1) > n_dim]
dirs_ = dirs[np.invert(np.isnan(pert_lengths)).sum(-1) > n_dim]
dists = np.zeros((len(images_), n_dim, n_samples))
for i, img in enumerate(images_):
    for n in np.arange(1, n_dim+1):
        dists[i, n-1] = get_dist_dec(img, labels_[i], dirs_[i,:n],
                                     model_natural, n_samples=n_samples)
np.savez(f'../data/distance_to_boundary_natural_{seed}.npz', data=dists)

images_ = images[np.invert(np.isnan(pert_lengths_madry)).sum(-1) > n_dim]
labels_ = labels[np.invert(np.isnan(pert_lengths_madry)).sum(-1) > n_dim]
dirs_ = dirs[np.invert(np.isnan(pert_lengths_madry)).sum(-1) > n_dim]
dists = np.zeros((len(images_), n_dim, n_samples))
for i, img in enumerate(images_):
    for n in np.arange(1, n_dim+1):
        dists[i, n-1] = get_dist_dec(img, labels_[i], dirs_[i,:n],
                                     model_robust, n_samples=n_samples)
np.savez(f'../data/distance_to_boundary_robust_{seed}.npz', data=dists)