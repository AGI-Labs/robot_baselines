import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _linear(in_dim, out_dim):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight.data, gain=1)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class VGGSoftmax(nn.Module):
    def __init__(self, bias=None):
        super().__init__()
        c1, a1 = nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c2, a2 = nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        m1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        c3, a3 = nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c4, a4 = nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        self.vgg = nn.Sequential(c1, a1, c2, a2, m1, c3, a3, c4, a4)
        self.extra_convs = nn.Conv2d(128, 128, 3, stride=2, padding=1)

    def forward(self, x):
        # vgg convs and 2D softmax
        x = self.vgg(x)
        x = self.extra_convs(x)
        B, C, H, W = x.shape
        x = F.softmax(x.view((B, C, H * W)), dim=2).view((B, C, H, W))

        # calculate and return expected keypoints
        h = torch.linspace(-1, 1, H).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 3)
        w = torch.linspace(-1, 1, W).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 2)
        return torch.cat([torch.sum(a, 2) for a in (h, w)], 1)


class PointPredictor(nn.Module):
    def __init__(self, feature_dim, bias=None, hidden_dim=16):
        super().__init__()
        fc1, a1 = nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True)
        fc2 = nn.Linear(hidden_dim, 3, bias=False)
        self.top = nn.Sequential(fc1, a1, fc2)
        bias = np.zeros(3).astype(np.float32) if bias is None else np.array(bias).reshape(3)
        self.register_parameter('bias', nn.Parameter(torch.from_numpy(bias).float(), requires_grad=True))
    
    def forward(self, x):
        return self.top(x) + self.bias


class CNNPolicy(nn.Module):
    def __init__(self, features, sdim=7, adim=7):
        super().__init__()
        self._features = features
        f1, a1 = _linear(256 + sdim, 128), nn.Tanh()
        f2 = _linear(128, adim)
        self._pi = nn.Sequential(f1, a1, f2)
    
    def forward(self, images, states):
        feat = self._features(images)
        return self._pi(torch.cat((feat, states), 1))


def restore_pretrain(model):
    # only import torchvision here since it's not in polymetis env 
    from torchvision import models
    if isinstance(model, VGGSoftmax):
        pt = models.vgg16(pretrained=True).features[:10]
        model.vgg.load_state_dict(pt.state_dict())
        return model
    raise NotImplementedError
