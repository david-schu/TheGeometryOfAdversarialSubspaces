import matplotlib.pyplot as plt
import numpy as np
import plots as pl
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

data = np.load('../data/cnn.npy', allow_pickle=True)
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']

# pl.show_orth(dirs[0])
pl.plot_advs()
