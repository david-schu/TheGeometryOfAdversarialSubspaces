import torch
from torchvision import datasets, transforms
from models import model, eval
from abs_models import models as mz
from utils import dev

# Initialize model and data loader
model_normal = model.madry()
model_normal.load_state_dict(torch.load('./models/natural.pt',map_location=dev()))
# model_adv = model.madry()
if torch.cuda.is_available():
    model_normal = model_normal.cuda()
    # model_adv = model_adv.cuda()
abs = mz.get_VAE(n_iter=50)
train_batch_size = 50
eval_batch_size = 200
learning_rate = 1e-4
epsilon = [1]

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                     transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])),
    batch_size=train_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                     transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])),
    batch_size=eval_batch_size)

nb_epochs = 2

# print("Training Model")
# model_normal.trainTorch(train_loader, nb_epochs, learning_rate)

# Evaluation
eval.evalClean(model_normal, test_loader)
eval.evalAdvAttack(model_normal, test_loader, epsilon=epsilon)

print("Training on Adversarial Samples")
# model_adv.advTrain(train_loader, nb_epochs, learning_rate, epsilon=epsilon)

# Evaluating Again
# eval.evalClean(model_adv, test_loader)
# eval.evalAdvAttack(model_adv, test_loader, epsilon=epsilon)

# torch.save(model_normal.state_dict(), 'models/natural_normalized.pt')
# torch.save(model_adv.state_dict(), 'models/adv_trained_l2.pt')
print('Done')
