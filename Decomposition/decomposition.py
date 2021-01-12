import matplotlib.pyplot as plt
import numpy as np
import plots as pl
from models import model
import torch
from utils import dev
import foolbox
import utils as u
from matplotlib.ticker import FormatStrFormatter
# data = np.load('../data/orth_const/orth_consts.npy',allow_pickle=True).item()

# pert_lengths = data['pert_lengths'][::2]
# orth_consts = data['orth_consts'][::2]
# plt.figure()
# for i, p in enumerate(pert_lengths):
#     mean_pert_lengths = np.mean(p[p[:,1]>0], axis=0)
#     plt.scatter([0,1],mean_pert_lengths, label=str(orth_consts[i]))
# plt.xlabel('Adversarial Dimension')
# plt.ylabel('l2 Perturbation Length')
# plt.legend()


data = np.load('../data/cnn.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']

model = model.madry()
model.load_state_dict(torch.load('./../models/normal.pt', map_location=torch.device(dev())))
model.eval()
fmodel = foolbox.models.PyTorchModel(model,
                                     bounds=(0., 1.),
                                     device=u.dev())

pl.plot_cw_surface(images[0],advs[0,0], advs[0,1], model)



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
