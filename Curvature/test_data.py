import sys

import dill
from tqdm import tqdm
import numpy as np
import torch
from torchvision import datasets

sys.path.insert(0, './..')
sys.path.insert(0, '../data')

from models import model as model_loader
from utils import dev, make_orth_basis
from robustness1.datasets import CIFAR
from curve_utils import *

sys.path.insert(0, './../..')

import response_contour_analysis.utils.dataset_generation as data_utils
import response_contour_analysis.utils.model_handling as model_utils
import response_contour_analysis.utils.principal_curvature as curve_utils

code_directory = '../../'

batch_size = 10
num_images = 50
num_advs = 10
seed = 0

num_iters = 2 # for paired image boundary search
num_steps_per_iter = 100 # for paired image boundary search
dtype = torch.double

# load data
data_natural = np.load(code_directory+'AdversarialDecomposition/data/cifar_natural_diff.npy', allow_pickle=True).item()
data_madry = np.load(code_directory+'AdversarialDecomposition/data/cifar_robust_diff.npy', allow_pickle=True).item()

# load models
ds = CIFAR(code_directory+'AdversarialDecomposition/data/cifar-10-batches-py')
classifier_model = ds.get_model('resnet50', False)

model_natural = model_loader.cifar_pretrained(classifier_model, ds)
resume_path = code_directory+'AdversarialDecomposition/models/nat_diff.pt'
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

model_madry = model_loader.cifar_pretrained(classifier_model, ds)
resume_path = code_directory+'AdversarialDecomposition/models/rob_diff.pt'
checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device(dev()))
state_dict_path = 'model'
if not ('model' in checkpoint):
    state_dict_path = 'state_dict'
sd = checkpoint[state_dict_path]
sd = {k[len('module.'):]:v for k,v in sd.items()}
model_madry.load_state_dict(sd)
model_madry.to(dev())
model_madry.double()
model_madry.eval()

model_data_zip = zip([model_natural, model_madry], [data_natural, data_madry], ['natural', 'robust'])
for model_, data_, name_ in model_data_zip:
    clean_image_splits = torch.split(torchify(data_['images']), batch_size, dim=0)
    clean_model_predictions = []
    for batch in clean_image_splits:
        clean_model_predictions.append(torch.argmax(model_(batch), dim=1).detach().cpu().numpy())
    clean_model_predictions = np.stack(clean_model_predictions, axis=0).reshape(-1)

    assert np.all(clean_model_predictions == data_['labels']), f'{name_} failed.'

    for image_idx in range(data_['images'].shape[0]):
        for adv_idx in range(data_['advs'][image_idx, ...].shape[0]):
            if np.isfinite(data_['pert_lengths'][image_idx, adv_idx]):
                adv_image = torchify(data_['advs'][image_idx, adv_idx, ...]).reshape(data_['images'][image_idx, ...].shape)[None, ...]
                adv_model_prediction = torch.argmax(model_(adv_image), dim=1).detach().cpu().numpy()
                assert adv_model_prediction != clean_model_predictions[image_idx], f'{name_} failed.'
                assert adv_model_prediction == data_['adv_class'][image_idx, adv_idx], f'{name} failed.'

print('test passed')