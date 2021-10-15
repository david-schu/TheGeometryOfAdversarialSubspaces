import sys
sys.path.insert(0, './..')
sys.path.insert(0, '../data')
from robustness import train, defaults
from cifar_models import model_utils
from robustness.datasets import CIFAR

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

# Hard-coded dataset, architecture, batch size, workers
ds = CIFAR('../data')
model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

# # Create a cox store for logging
# out_store = cox.store.Store('./nat')
#
# # Hard-coded base parameters
# train_kwargs = {
#     'out_dir': "./nat/train_out",
#     'adv_train': 0,
# }
# train_args = Parameters(train_kwargs)
#
# # Fill whatever parameters are missing from the defaults
# train_args = defaults.check_and_fill_args(train_args, defaults.CONFIG_ARGS, CIFAR)
# train_args = defaults.check_and_fill_args(train_args,
#                         defaults.TRAINING_ARGS, CIFAR)
#
#
# # Train a model
# train.train_model(train_args, model, (train_loader, val_loader), store=out_store)


# Create a cox store for logging
out_store = cox.store.Store('./rob_new')

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "./rob/train_out",
    'adv_train': 1,
    'constraint': '2',
    'eps': 0.5,
    'epochs': 200,
    'custom_lr_multplier': [(100,.1), (150,.1)],
    'weight_decay': 5e-4,
    'attack_lr': 1.5,
    'attack_steps': 20
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args, defaults.CONFIG_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args, defaults.TRAINING_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args, defaults.PGD_ARGS, CIFAR)

# Train a model
train.train_model(train_args, model, (train_loader, val_loader), store=out_store)
