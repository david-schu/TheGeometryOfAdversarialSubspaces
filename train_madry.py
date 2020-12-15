import torch
from torchvision import datasets, transforms
from models import model, eval

# Initialize model and data loader
model_normal = model.madry()
model_adv = model.madry()
if torch.cuda.is_available():
    model_normal = model_normal.cuda()
    model_adv = model_adv.cuda()
nb_epochs = 15
batch_size = 128
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./../data', train=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size)

print("Training Model")
model_normal.trainTorch(train_loader, nb_epochs, learning_rate)

# Evaluation
eval.evalClean(model_normal, test_loader)
eval.evalAdvAttack(model_normal, test_loader)

print("Training on Adversarial Samples")
model_adv.advTrain(train_loader, nb_epochs, learning_rate)

# Evaluating Again
eval.evalClean(model_adv, test_loader)
eval.evalAdvAttack(model_adv, test_loader)

torch.save(model_normal.state_dict(), 'models/normal.pt')
torch.save(model_adv.state_dict(), 'models/adv_trained.pt')
print('Done')
