import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
from abs_models import utils as u
import foolbox
from foolbox import attacks as fa

NB_EPOCHS = 4
BATCH_SIZE = 128
LEARNING_RATE = .001


# Creating a simple network
class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)

    # Normal Training
    def trainTorch(self,
                   train_loader,
                   nb_epochs=NB_EPOCHS,
                   learning_rate=LEARNING_RATE
                   ):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = []
        total = 0
        correct = 0
        step = 0
        for _epoch in range(nb_epochs):
            for xs, ys in train_loader:
                xs, ys = Variable(xs), Variable(ys)
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()
                optimizer.zero_grad()
                preds = self(xs)
                # print("HI")
                loss = F.nll_loss(preds, ys)
                # print("HADSFSDF")
                loss.backward()  # calc gradients
                train_loss.append(loss.data.item())
                optimizer.step()  # update gradients

                preds_np = preds.cpu().detach().numpy()
                correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
                total += train_loader.batch_size
                step += 1
                if total % 1000 == 0:
                    acc = float(correct) / total
                    print('[%s] Training accuracy: %.2f%%' % (step, acc * 100))
                    total = 0
                    correct = 0

    # Adversarial Training
    def advTrain(self,
                 train_loader,
                 nb_epochs=NB_EPOCHS,
                 learning_rate=LEARNING_RATE
                 ):
        self.trainTorch(train_loader, nb_epochs, learning_rate)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = []
        total = 0
        correct = 0
        totalAdv = 0
        correctAdv = 0
        step = 0
        # breakstep = 0
        for _epoch in range(nb_epochs):
            for xs, ys in train_loader:
                # Normal Training
                xs, ys = Variable(xs), Variable(ys)
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()
                optimizer.zero_grad()
                preds = self(xs)
                loss = F.nll_loss(preds, ys)
                loss.backward()  # calc gradients
                train_loss.append(loss.data.item())
                optimizer.step()  # update gradients
                preds_np = preds.cpu().detach().numpy()
                correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
                total += train_loader.batch_size

                # Adversarial Training
                self.eval()
                fmodel = foolbox.models.PyTorchModel(self,  # return logits in shape (bs, n_classes)
                                                     bounds=(0., 1.),  # num_classes=10,
                                                     device=u.dev())
                attack = fa.LinfProjectedGradientDescentAttack(rel_stepsize=0.01 / 0.3,
                                                               steps=100,
                                                               random_start=True, )
                xs, _, success = attack(fmodel, xs, ys, epsilons=[0.3])

                xs, ys = Variable(xs[0]), Variable(ys)
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()
                optimizer.zero_grad()
                preds = self(xs)
                loss = F.nll_loss(preds, ys)
                loss.backward()  # calc gradients
                train_loss.append(loss.data.item())
                optimizer.step()  # update gradients
                preds_np = preds.cpu().detach().numpy()
                correctAdv += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
                totalAdv += train_loader.batch_size

                step += 1
                if total % 1000 == 0:
                    acc = float(correct) / total
                    print('[%s] Clean Training accuracy: %.2f%%' % (step, acc * 100))
                    total = 0
                    correct = 0
                    accAdv = float(correctAdv) / totalAdv
                    print('[%s] Adv Training accuracy: %.2f%%' % (step, accAdv * 100))
                    totalAdv = 0
                    correctAdv = 0

class madry(torch.nn.Module):
    def __init__(self):
        super(madry, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding=2)
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxPool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxPool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def trainTorch(self,
                   train_loader,
                   nb_epochs=NB_EPOCHS,
                   learning_rate=LEARNING_RATE
                   ):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = []
        total = 0
        correct = 0
        step = 0
        for _epoch in range(nb_epochs):
            for xs, ys in train_loader:
                xs, ys = Variable(xs), Variable(ys)
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()
                optimizer.zero_grad()
                preds = self(xs)
                loss = F.nll_loss(preds, ys)
                loss.backward()  # calc gradients
                train_loss.append(loss.data.item())
                optimizer.step()  # update gradients

                preds_np = preds.cpu().detach().numpy()
                correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
                total += train_loader.batch_size
                step += 1
                if total % 1000 == 0:
                    acc = float(correct) / total
                    print('[%s] Training accuracy: %.2f%%' % (step, acc * 100))
                    total = 0
                    correct = 0
