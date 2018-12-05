# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='SpaceInvaders-v0', type=str, help='gym environment')
    parser.add_argument('--agent_file', default='SpaceInvaders-v0.fskip7.160.tar', type=str)
    parser.add_argument('--ae_file', default='AE', type=str)
    parser.add_argument('--latent_size', default='64', type=int)
    return parser.parse_args()

#prepro = lambda img: imresize(img[35:195], (80,80)).astype(np.float32).mean(2).reshape(1,80,80)/255.
class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 5 * 5, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs)) ; x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x)) ; x = F.elu(self.conv4(x))
        x = self.linear(x.view(-1, 32 * 5 * 5))
        return self.critic_linear(x), self.actor_linear(x)

class AutoEncoder(nn.Module):
    def __init__(self, latent_size):
        super(AutoEncoder, self).__init__()
        self.hidden_units = 3 * 3 * 256
        self.latent_size = latent_size
        self.conv1 = (nn.Conv2d(4, 64, 3, stride=2, padding=1))
        self.conv2 = (nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.conv3 = (nn.Conv2d(128, 256, 4, stride=2, padding=1))
        self.conv4 = (nn.Conv2d(256, 256, 4, stride=2, padding=1))
        self.conv5 = (nn.Conv2d(256, 256, 3, stride=1, padding=0))
        self.fc_e = nn.Linear(self.hidden_units, self.latent_size)
        self.fc_g = nn.Linear(self.latent_size, self.latent_size)
        self.deconv1 = nn.ConvTranspose2d(64, 512, 4, stride=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=0) # 10
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) #20
        self.deconv4 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1) #40
        self.deconv5 = nn.ConvTranspose2d(128, 4, 4, stride=2, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view((-1, self.hidden_units))
        x = self.fc_e(x)
        x = self.fc_g(x)
        x = F.relu(x)
        x = x[0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return x



args = get_args()

print("loading models")
model = NNPolicy(channels=4, num_actions=6) # init a local (unshared) model
AE = AutoEncoder(args.latent_size)
#enc = Encoder()
#gen = Generator()
model.load_state_dict(torch.load(args.agent_file))
#AE.load_state_dict(torch.load(args.ae_file))
#gen.load_state_dict(torch.load(args.gen_file))


dummy_input1 = Variable(torch.randn(1, 4, 80, 80))
dummy_input2 = Variable(torch.randn(1, 4, 80, 80))

print("saving models")
torch.onnx.export(model, dummy_input1, args.agent_file+".onnx")
torch.onnx.export(AE, dummy_input2, "AutoEncoder.onnx")



