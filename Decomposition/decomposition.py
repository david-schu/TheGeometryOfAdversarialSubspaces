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


data = np.load('../data/cnn.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']
#
# # pert_lengths=pert_lengths[:, :7]
# pert_lengths[pert_lengths==0] = np.nan
# pert_lengths_mean = np.nanmean(pert_lengths,axis=0)
# pert_lengths_var = np.nanstd(pert_lengths, axis=0)
# plt.errorbar(np.arange(len(pert_lengths_mean))+1, pert_lengths_mean, pert_lengths_var, fmt='o', alpha=0.7)

### madry
data = np.load('../data/madry.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']

# pert_lengths=pert_lengths[:, :7]n
# pert_lengths[pert_lengths==0] = np.nan
# pert_lengths_mean = np.nanmean(pert_lengths,axis=0)
# pert_lengths_var = np.nanstd(pert_lengths, axis=0)
# plt.errorbar(np.arange(len(pert_lengths_mean))+1, pert_lengths_mean, pert_lengths_var, fmt='o', alpha=0.7)
#
# plt.title('Perturbation length of first ' + str(pert_lengths.shape[-1]) + ' adversarial directions')
# plt.xlabel('n')
# plt.xticks(np.arange(len(pert_lengths_mean))+1)
# plt.ylabel('adversarial vector length ($l2-norm$)')
# plt.ylim(0)
# plt.legend(["CNN - no adversarial training", "CNN - with adversarial training"], loc='lower right')
# plt.show()

### ABS
data = np.load('../data/abs.npy', allow_pickle=True).item()
advs = data['advs']
pert_lengths = data['pert_lengths']
classes = data['adv_class']
dirs = data['dirs']
images = data['images']
labels = data['labels']

# pert_lengths=pert_lengths[:, :7]
pert_lengths[pert_lengths==0] = np.nan
pert_lengths_mean = np.nanmean(pert_lengths,axis=0)
pert_lengths_var = np.nanstd(pert_lengths, axis=0)
plt.errorbar(np.arange(len(pert_lengths_mean))+1, pert_lengths_mean, pert_lengths_var, fmt='o')


plt.title('Perturbation length of first ' + str(pert_lengths.shape[-1]) + ' adversarial directions')
plt.xlabel('n')
plt.xticks(np.arange(len(pert_lengths_mean))+1)
plt.ylabel('adversarial vector length ($l2-norm$)')
plt.ylim(0)

plt.legend(["ABS"])
plt.show()
print('Done')


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
