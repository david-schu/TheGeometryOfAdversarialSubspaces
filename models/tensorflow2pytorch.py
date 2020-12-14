import torch
from torchvision import datasets, transforms
from models import model as md, eval
from madry.mnist_challenge.model import Model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
model = Model()
checkpoint = tf.train.latest_checkpoint('./../AnalysisBySynthesis/madry/mnist_challenge/models/adv_trained')
restorer = tf.train.Saver()
restorer.restore(sess, checkpoint)


model_madry = md.madry()
weights_cv1 = torch.from_numpy(sess.run('Variable:0')).permute((3, 2, 0, 1))
bias_cv1 = torch.from_numpy(sess.run('Variable_1:0'))
model_madry.conv1.weight = torch.nn.Parameter(weights_cv1)
model_madry.conv1.bias = torch.nn.Parameter(bias_cv1)

weights_cv2 = torch.from_numpy(sess.run('Variable_2:0')).permute((3, 2, 0, 1))
bias_cv2 = torch.from_numpy(sess.run('Variable_3:0'))
model_madry.conv2.weight = torch.nn.Parameter(weights_cv2)
model_madry.conv2.bias = torch.nn.Parameter(bias_cv2)

weights_fc1 = torch.from_numpy(sess.run('Variable_4:0')).permute((1,0))
bias_fc1 = torch.from_numpy(sess.run('Variable_5:0'))
model_madry.fc1.weight = torch.nn.Parameter(weights_fc1)
model_madry.fc1.bias = torch.nn.Parameter(bias_fc1)

weights_fc2 = torch.from_numpy(sess.run('Variable_6:0')).permute((1,0))
bias_fc2 = torch.from_numpy(sess.run('Variable_7:0'))
model_madry.fc2.weight = torch.nn.Parameter(weights_fc2)
model_madry.fc2.bias = torch.nn.Parameter(bias_fc2)

batch_size = 128
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./../data', train=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size)

eval.evalClean(model_madry, test_loader)
print('Done!')