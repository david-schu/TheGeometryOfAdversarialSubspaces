import sys
sys.path.insert(0, './../')
sys.path.insert(0,'../../foolbox/examples/zoo/mnist')

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
from random import randint
from matplotlib import pyplot as plt
import seaborn as sns
import foolbox
from foolbox import attacks as fa
from foolbox.distances import LpDistance

# own modules
from abs_models import utils as u
from abs_models import models as mz
from abs_models import attack_utils as au


def orth_check(adv_dirs):
    orth = np.eye(len(adv_dirs))
    for i in range(len(adv_dirs)):
        for j in range(i + 1, len(adv_dirs)):
            orth[i, j] = np.dot(adv_dirs[i].T, adv_dirs[j])
            orth[j, i] = orth[i, j]

    return orth


def classification(img, model):
    pred = model(img).detach().numpy()
    img_class = np.argmax(pred,axis=-1)
    return img_class

def dirs_to_attack_format(dirs):
    max_dim = max(len(elem) for elem in dirs)
    attack_dirs=np.zeros([len(dirs),max_dim,dirs[0].shape[-1]])
    for i,dir in enumerate(dirs):
        attack_dirs[i,:len(dir)] = dir
    return torch.tensor(attack_dirs)


class attack_orth(fa.base.MinimizationAttack):
    def __init__(self,input_attack,params, adv_dirs=[],orth_const=50):
        super(attack_orth,self).__init__()
        self.input_attack = input_attack(**params)
        self.distance = LpDistance(2)
        self.dirs = adv_dirs
        self.orth_const = orth_const


    def run(self,model,inputs,criterion,**kwargs):
        return self.input_attack.run(model,inputs,criterion,dirs=self.dirs,orth_const=self.orth_const, **kwargs)
    def distance(self):
        ...


# model = mz.get_VAE(n_iter=10)              # ABS, do n_iter=50 for original model
# model = mz.get_VAE(binary=True)           # ABS with scaling and binaryzation
# model = mz.get_binary_CNN()               # Binary CNN
model = mz.get_CNN()                      # Vanilla CNN
# model = mz.get_NearestNeighbor()          # Nearest Neighbor, "nearest L2 dist to each class"=logits
# model = mz.get_madry()                    # Robust network from Madry et al. in tf
# model = create()
model.eval()
fmodel = foolbox.models.PyTorchModel(model,   # return logits in shape (bs, n_classes)
                                     bounds=(0., 1.), #num_classes=10,
                                     device=u.dev())

# images, labels = foolbox.utils.samples(fmodel, dataset="mnist", batchsize=2)  # returns random batch as torch tensor
# # rand = randint(0,19)
# # images = images[rand].unsqueeze(0)
# # labels = labels[rand].unsqueeze(0)
# labels = labels.long()

images = torch.load('images.pt')
labels = torch.load('labels.pt')

# user initialization
n_adv_dims = 1
max_runs = 10
show_plots = False
early_stop = 3
norm_order = 2
steps = 500
input_attack = fa.L2CarliniWagnerAttack
epsilons = [None]

# variable initializations
attack_params = {
        'binary_search_steps':9,
        'initial_const':1e-2,
        'steps':steps,
        'confidence':1,
        'abort_early':True
    }

n_images = len(images)
n_pixel = images.shape[-1]**2
x_orig = u.t2n(images).reshape([n_images,n_pixel])

orth_consts = [50]

for orth_const in orth_consts:
    count = 0
    min_dim = 0
    adv_dirs = []
    pert_lengths = []
    advs = []
    dirs = torch.tensor([])
    adv_dirs =[]
    adv_class = []

    for run in range(max_runs):
        print('Run %d - Adversarial Dimension %d...' % (run+1, min_dim + 1))

        attack = attack_orth(input_attack=input_attack, params=attack_params,adv_dirs=dirs,orth_const=orth_const)
        adv, _, success = attack(fmodel, images, labels, epsilons=epsilons)

        # check if adversarials were found and stop early if not
        if success.sum() == 0:
            print('--No attack within bounds found--')
            count +=1
            if early_stop == count:
                print('No more adversarials found ----> early stop!')
                break
            continue

        count = 0

        classes = classification(adv[0],model)
        min_dim = n_adv_dims
        for i, a in enumerate(adv[0]):
            a_ = a.flatten().numpy()
            pert_length = np.linalg.norm(a_ - x_orig[i], ord=2)
            if run == 0:
                min_dim = 1
                advs.append(np.array([a_]))
                pert_lengths.append(np.array([pert_length]))
                adv_dirs.append(np.array([(a_ - x_orig[i])/pert_length]))
                adv_class.append(np.array([classes[i]]))
            else:
                dim = np.sum(pert_lengths[i] < pert_length)
                min_dim = np.minimum(min_dim,dim)+1
                advs[i] = np.vstack([advs[i][:dim], a_])
                adv_dirs[i] = np.vstack([adv_dirs[i][:dim], (a_ - x_orig[i])/pert_length])
                adv_class[i] = np.append(adv_class[i][:dim], classes[i])
                pert_lengths[i] = np.append(pert_lengths[i][:dim], pert_length)
        dirs = dirs_to_attack_format(adv_dirs)
        if min_dim == n_adv_dims:
            break
data = {
    "advs": advs,
    "adv_dirs": adv_dirs,
    "adv_class": adv_class,
    "pert_lengths": pert_lengths,
    "images": images,
    "labels": labels
}

np.save('temp.npy', data)