import numpy as np
from models import model
import torch
from robustness.datasets import CIFAR
import dill

from utils import dev

import tqdm


def get_dist_dec(orig, label, dirs, model, min_dist=.1, n_samples=1000, return_angles=False):
    shape = orig.shape
    n_steps = 20
    n_dirs = len(dirs)
    dirs = dirs.reshape((n_dirs, -1))

    upper = np.full((n_samples, 1), np.inf)
    lower = np.ones((n_samples, 1)) * min_dist

    scales = np.ones((n_samples, 1)) * min_dist

    coeffs = abs(np.random.normal(size=[n_samples, n_dirs]))
    sample_dirs = (coeffs @ dirs)
    sample_dirs = sample_dirs / np.linalg.norm(sample_dirs, axis=-1, keepdims=True)

    dists = np.full(n_samples, np.nan)

    for i in range(n_steps):
        input_dirs = scales * sample_dirs
        input_ = (input_dirs + orig.flatten()[None])
        input = torch.split(torch.tensor(input_.reshape((-1,) + shape), device=dev()), 100)

        preds = np.empty(0)
        for batch in input:
            preds = np.concatenate((preds, model(batch).argmax(-1).cpu().numpy()), axis=0)

        is_adv = np.invert(preds == label)
        dists[is_adv] = scales[is_adv, 0]

        upper[is_adv] = scales[is_adv]
        lower[~is_adv] = scales[~is_adv]
        scales[is_adv] = (upper[is_adv] + lower[is_adv]) / 2
        scales[~is_adv] = lower[~is_adv] * 2

        in_bounds = np.logical_or(input_.max(-1) <= 1, input_.min(-1) >= 0)
        dists[~in_bounds] = np.nan

    if return_angles:
        angles = np.arccos((sample_dirs@dirs.T).clip(-1,1)).min(-1)
        angles[np.isnan(dists)] = np.nan
        return dists, angles
    return dists


# Load models

ds = CIFAR('./data/cifar-10-batches-py')
classifier_model = ds.get_model('resnet50', False)
model_natural = model.cifar_pretrained(classifier_model, ds)

resume_path = './models/cifar_models/cifar_nat.pt'
checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))

state_dict_path = 'model'
if not ('model' in checkpoint):
    state_dict_path = 'state_dict'
sd = checkpoint[state_dict_path]
sd = {k[len('module.'):]: v for k, v in sd.items()}
model_natural.load_state_dict(sd)
model_natural.to(dev())
model_natural.double()
model_natural.eval()

classifier_model = ds.get_model('resnet50', False)
model_robust = model.cifar_pretrained(classifier_model, ds)

resume_path = './models/cifar_models/cifar_l2_0_5.pt'
checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))

state_dict_path = 'model'
if not ('model' in checkpoint):
    state_dict_path = 'state_dict'
sd = checkpoint[state_dict_path]
sd = {k[len('module.'):]: v for k, v in sd.items()}
model_robust.load_state_dict(sd)
model_robust.to(dev())
model_robust.double()
model_robust.eval()

# load data
data_nat = np.load('./data/cifar_natural.npy', allow_pickle=True).item()
advs = data_nat['advs']
pert_lengths = data_nat['pert_lengths']
classes = data_nat['adv_class']
dirs = data_nat['dirs']
images = data_nat['images']
labels = data_nat['labels']
pert_lengths = data_nat['pert_lengths']

data_madry = np.load('./data/cifar_robust.npy', allow_pickle=True).item()
advs_madry = data_madry['advs']
pert_lengths_madry = data_madry['pert_lengths']
classes_madry = data_madry['adv_class']
dirs_madry = data_madry['dirs']

n_samples = 100
n_dims = 50

# img_indices = np.random.choice(np.arange(100), size=n_images, replace=False)
img_indices = np.array([18, 36, 67, 88, 92])

#natural
images_ = images#[img_indices]
labels_ = labels#[img_indices]
dirs_nat = dirs#[img_indices]
dirs_rob = dirs_madry#[img_indices]
min_dist_nat=pert_lengths[:, 0]
min_dist_rob=pert_lengths_madry[:, 0]

dists_natural = np.zeros((len(images_), n_dims, n_samples))
dists_robust = np.zeros((len(images_), n_dims, n_samples))
angles_natural = np.zeros((len(images_), n_dims, n_samples))
angles_robust = np.zeros((len(images_), n_dims, n_samples))

for i, img in enumerate(tqdm.tqdm(images_)):
    for n in np.arange(1, n_dims+1):
        dists_natural[i, n-1], angles_natural[i, n-1] = get_dist_dec(img, labels_[i], dirs_nat[i, :n], model_natural,
                                            min_dist=0.5 * min_dist_nat[i], n_samples=n_samples, return_angles=True)

        dists_robust[i, n - 1], angles_robust[i, n-1] = get_dist_dec(img, labels_[i], dirs_rob[i, :n], model_robust,
                                            min_dist=0.5 * min_dist_rob[i], n_samples=n_samples, return_angles=True)
        data = {
            'dists_natural': dists_natural,
            'dists_robust': dists_robust,
            'angles_natural': angles_natural,
            'angles_robust': angles_robust
        }
        save_path = './data/dists_to_bnd.npy'
        np.save(save_path, data)
