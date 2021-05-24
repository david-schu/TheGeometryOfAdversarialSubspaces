import numpy as np
import UniversalPerturbations.deepfool as deepfool
import torch
from torchvision import transforms, datasets
from utils import dev

from models import model

def project_perturbation(xi, p, perturbation):
    if p == 2:
        perturbation = perturbation * np.minimum(1, xi / np.linalg.norm(perturbation))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), xi)
    return perturbation


def generate(trainset, testset, net, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.2, max_iter_df=20):
    '''
    :param trainset: Pytorch Dataloader with train data
    :param testset: Pytorch Dataloader with test data
    :param net: Network to be fooled by the adversarial examples
    :param delta: 1-delta represents the fooling_rate, and the objective
    :param max_iter_uni: Maximum number of iterations of the main algorithm
    :param p: Only p==2 or p==infinity are supported
    :param num_class: Number of classes on the dataset
    :param overshoot: Parameter to the Deep_fool algorithm
    :param max_iter_df: Maximum iterations of the deep fool algorithm
    :return: perturbation found (not always the same on every run of the algorithm)
    '''
    net.to(dev())
    net.eval()
    print('Device: ' + str(dev()))

    # Importing images and creating an array with them
    img_trn = []
    for batch in trainset:
        for image in batch[0]:
            img_trn.append(image.numpy())
    img_trn = np.array(img_trn)


    # Finding labels for original images
    orig_labels_test = torch.tensor(np.zeros(0, dtype=np.int64))
    correct = 0
    n_test_samples = 0
    for inputs, labels in testset:
        inputs, labels = inputs.to(dev()), labels.to(dev())
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        orig_labels_test = torch.cat((orig_labels_test, predicted.cpu()))
        correct += (predicted == labels).sum()
        n_test_samples += len(inputs)

    accuracy = correct/n_test_samples
    print('Clean model accuracy: %.2f%%' % (accuracy*100))

    # Setting the number of images to 300  (A much lower number than the total number of instances on the training set)
    # To verify the generalization power of the approach
    num_img_trn = 100
    img_indices = np.arange(len(img_trn))
    np.random.shuffle(img_indices)
    img_indices = img_indices[: num_img_trn]

    inputs = torch.tensor(img_trn[img_indices].astype('float32'), device=dev())

    _, orig_labels_train = net(inputs).max(1)
    orig_labels_train = orig_labels_train.cpu()

    # Initializing the perturbation to 0s
    v = np.zeros((28, 28))

    #Initializing fooling rate and iteration count
    fooling_rate = 0.0
    iter = 0

    fooling_rates=[0]
    accuracies = [accuracy]
    total_iterations = [0]
    # Begin of the main loop on Universal Adversarial Perturbations algorithm
    while fooling_rate < 1-delta and iter < max_iter_uni:
        print("Iteration  ", iter+1)

        for j, idx in enumerate(img_indices):
            # Generating the original image from data
            cur_img = img_trn[idx][0]
            orig_label = orig_labels_train[j]

            # Generating a perturbed image from the current perturbation v and the original image
            per_img = np.clip(cur_img+v,0,1)
            per_img1 = torch.tensor(per_img.astype('float32'), device=dev()).reshape(-1, 1, 28, 28)

            # Feeding the perturbed image to the network and storing the label returned
            _, per_label = net(per_img1).max(1)
            per_label = per_label.cpu()

            # If the label of both images is the same, the perturbation v needs to be updated
            if orig_label == per_label:
                # Finding a new minimal perturbation with deepfool to fool the network on this image
                dr, iter_k, label, k_i, pert_image = deepfool.deepfool(per_img1[0], net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                if iter_k < max_iter_df-1:

                    v[:, :] += dr[0, 0, :, :]
                    v = project_perturbation(xi, p, v)

        iter += 1

        with torch.no_grad():
            # Compute fooling_rate
            per_labels_test = torch.tensor(np.zeros(0, dtype=np.int64))
            correct = 0
            # Finding labels for perturbed images
            for batch_index, (inputs, labels) in enumerate(testset):
                inputs = (inputs+v.astype('float32'))
                inputs, labels = inputs.to(dev()), labels.to(dev())
                outputs = net(inputs.clip(0, 1))
                _, predicted = outputs.max(1)
                per_labels_test = torch.cat((per_labels_test, predicted.cpu()))
                correct += (predicted == labels).sum()
            torch.cuda.empty_cache()

            # Calculating the fooling rate by dividing the number of fooled images by the total number of images
            fooling_rate = float(torch.sum(orig_labels_test != per_labels_test))/n_test_samples

            print("FOOLING RATE: ", fooling_rate)
            print()

            fooling_rates.append(fooling_rate)
            accuracies.append(correct / n_test_samples)

    return v, fooling_rates, accuracies, iter
