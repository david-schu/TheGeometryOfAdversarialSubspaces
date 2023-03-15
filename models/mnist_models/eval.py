import torch
from torch.autograd import Variable
import numpy as np
import foolbox
from foolbox import attacks as fa
import utils as u


# Evaluate results on clean data
def evalClean(model=None, test_loader=None):
    print("Evaluating single model results on clean data")
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for xs, ys in test_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            preds1 = model(xs)
            preds_np1 = preds1.cpu().detach().numpy()
            finalPred = np.argmax(preds_np1, axis=1)
            correct += (finalPred == ys.cpu().detach().numpy()).sum()
            total += len(xs)
    acc = float(correct) / total
    print('Clean accuracy: %.2f%%' % (acc * 100))
    return acc

# Evaluate results on adversarially perturbed
def evalAdvAttack(model=None, test_loader=None, epsilon=[0.3]):
    print("Evaluating single model results on adv data")
    total = 0
    correct = 0
    model.eval()
    for xs, ys in test_loader:
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        # pytorch fast gradient method
        model.eval()
        fmodel = foolbox.models.PyTorchModel(model,  # return logits in shape (bs, n_classes)
                                             bounds=(0., 1.),  # num_classes=10,
                                             device=u.dev())
        attack = fa.L2ProjectedGradientDescentAttack(abs_stepsize=0.1,
                                                       steps=100,
                                                       random_start=True)
        adv, _, success = attack(fmodel, xs, ys, epsilons=epsilon)
        adv, ys = Variable(adv[0]), Variable(ys)
        preds1 = model(adv)
        preds_np1 = preds1.cpu().detach().numpy()
        finalPred = np.argmax(preds_np1, axis=1)
        correct += (finalPred == ys.cpu().detach().numpy()).sum()
        total += test_loader.batch_size
    acc = float(correct) / total
    print('Adv accuracy: %.2f%%' % (acc * 100))
    return acc