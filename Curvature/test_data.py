import sys

import dill
import numpy as np
import torch

sys.path.insert(0, './..')
sys.path.insert(0, '../data')

from models import model as model_loader
from utils import dev
from robustness1.datasets import CIFAR
from curve_utils import torchify, load_mnist, load_cifar

code_directory = '../../'

batch_size = 10

def run_test(model_data_zip):
    for model_, data_, name_ in model_data_zip:
        clean_image_splits = torch.split(torchify(data_['images']), batch_size, dim=0)
        clean_model_predictions = []
        for batch in clean_image_splits:
            clean_model_predictions.append(torch.argmax(model_(batch), dim=1).detach().cpu().numpy())
        clean_model_predictions = np.stack(clean_model_predictions, axis=0).reshape(-1)
        assert np.all(clean_model_predictions == data_['labels']), f'{name_} failed.'

        for image_idx in range(data_['images'].shape[0]):
            valid_advs = data_['advs'][image_idx, np.argwhere(np.isfinite(data_['pert_lengths'][image_idx, :]))]
            num_valid_advs = valid_advs.shape[0]
            adv_images = torchify(valid_advs).reshape((num_valid_advs, )+data_['images'][image_idx, ...].shape)
            adv_model_predictions = torch.argmax(model_(adv_images), dim=1).detach().cpu().numpy()
            assert np.all(adv_model_predictions != clean_model_predictions[image_idx]), f'{name_} failed.'
            assert np.all(adv_model_predictions == data_['adv_class'][image_idx, :num_valid_advs]), f'{name} failed.'

        print(f'{name_} test passed')


######
#MNIST
######
seed = 0
model_natural, data_natural, model_madry, data_madry = load_mnist(code_directory, seed)
model_data_zip = zip([model_natural, model_madry], [data_natural, data_madry], ['natural', 'robust'])
print('MNIST:')
run_test(model_data_zip)

######
#CIFAR
######
model_natural, data_natural, model_madry, data_madry = load_cifar(code_directory)
model_data_zip = zip([model_natural, model_madry], [data_natural, data_madry], ['natural', 'robust'])
print('CIFAR:')
run_test(model_data_zip)