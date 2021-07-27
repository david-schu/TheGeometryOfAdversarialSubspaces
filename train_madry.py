import torch
from torchvision import datasets, transforms
from models import model, eval

train_batch_size = 50
eval_batch_size = 200
learning_rate = 1e-3
epsilon = [2]
nb_epochs = 1

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=train_batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=eval_batch_size)

seeds = [12, 69, 420, 1202, 3000]

for i, seed in enumerate(seeds):

    torch.manual_seed(seed)

    # Initialize model and data loader
    model_nat = model.madry_diff()
    model_robust = model.madry_diff()
    if torch.cuda.is_available():
        model_nat = model_nat.cuda()
        model_robust = model_robust.cuda()

    print("Training Model")
    model_nat.trainTorch(train_loader, nb_epochs, learning_rate)

    # Evaluation
    eval.evalClean(model_nat, test_loader)
    eval.evalAdvAttack(model_nat, train_loader, epsilon=epsilon)
    torch.save(model_nat.state_dict(), 'models/natural_' + str(i) + '.pt')

    print("Training on Adversarial Samples")
    model_robust.advTrain(train_loader, nb_epochs, learning_rate, epsilon=epsilon)

    # Evaluating Again
    eval.evalClean(model_robust, test_loader)
    eval.evalAdvAttack(model_robust, test_loader, epsilon=epsilon)

    torch.save(model_robust.state_dict(), 'models/robust_' + str(i) + '.pt')

    del model_robust, model_nat
    print('Done')

