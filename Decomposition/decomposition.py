import matplotlib.pyplot as plt
import numpy as np
import plots as pl
from models import model
import torch
from utils import dev
from abs_models import models as mz
from matplotlib.ticker import FormatStrFormatter



data = np.load('../data/cnn.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']

pl.plot_var_hist(classes,labels,'Natural CNN - 5 Adv. Directions')

model_natural = model.madry()
model_madry = model.madry()
model_abs = mz.get_ABS(n_iter=50)
model_natural.load_state_dict(torch.load('./../models/natural.pt', map_location=torch.device(dev())))
model_madry.load_state_dict(torch.load('./../models/adv_trained.pt', map_location=torch.device(dev())))

mean_pert_length = np.mean(pert_lengths, axis=0)
dist_to_mean = np.sum(np.abs(pert_lengths - mean_pert_length), axis=-1)
min_idx = np.argmin(dist_to_mean)
print(classes[min_idx])
for i in range(1,5):
    pl.plot_cw_surface(images[min_idx], advs[min_idx,0], advs[min_idx,i], model_natural)
    pl.plot_cw_surface(images[min_idx], advs[min_idx, 0], advs[min_idx, i], model_madry)
    pl.plot_cw_surface(images[min_idx], advs[min_idx, 0], advs[min_idx, i], model_abs)



classes[pert_lengths==0] = np.nan
pl.plot_label_heatmap(classes, labels, show_all=False)



cnn = np.load('../data/cnn.npy', allow_pickle=True).item()
madry = np.load('../data/madry.npy', allow_pickle=True).item()
abs = np.load('../data/abs.npy', allow_pickle=True).item()


for i in range(5):
    plt.subplot(2,5,1+i)
    plt.title('Adv. class ' + str(cnn['adv_class'][i,0]))
    plt.imshow(np.reshape(cnn['advs'][i,0], [28,28]), cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.ylabel("Natural CNN")

    plt.subplot(2,5,6+i)
    plt.title('Adv. class ' + str(madry['adv_class'][i,0]))
    plt.imshow(np.reshape(madry['advs'][i,0], [28,28]), cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.ylabel("Madry CNN")
plt.suptitle('Adversarials of non-robust and robust models')
plt.show()
