from config import *
import torch.nn as nn
import cached_conv as cc
from torch.nn.utils import weight_norm

def GAN_module(x_raw, y_raw, current_epoch):  
    #x_raw: x
    #y_raw: x_hat

    if current_epoch > WARM_UP_EPOCH:
        xy = torch.cat([x_raw, y_raw], 0)

        discriminator = MultiScaleDiscriminator(DISC_NUM)
        features = discriminator(xy)

        feature_real, feature_fake = split_features(features)

        loss_dis = 0
        loss_adv = 0

        pred_real = 0
        pred_fake = 0

        feature_matching_fun = mean_difference
        for scale_real, scale_fake in zip(feature_real, feature_fake):
            current_feature_distance = sum(
                map(
                    feature_matching_fun,
                    scale_real[:],
                    scale_fake[:],
                )) / len(scale_real[:])

            feature_matching_distance = feature_matching_distance + current_feature_distance

            _dis, _adv = hinge_gan(scale_real[-1], scale_fake[-1])

            pred_real = pred_real + scale_real[-1].mean()
            pred_fake = pred_fake + scale_fake[-1].mean()

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv

        feature_matching_distance = feature_matching_distance / len(
            feature_real)
        
        loss_gen = {}
        loss_gen['feature_matching'] = LAMBDA_FEATURE_MATCHING * feature_matching_distance
        loss_gen['adversarial'] = LAMBDA_ADVERSARIAL * loss_adv

        return loss_dis, loss_gen

    else:
        return 0, None
    
        
class MultiScaleDiscriminator(nn.Module):

    def __init__(self, n_discriminators, n_channels=1) -> None:
        super().__init__()
        layers = []
        for i in range(n_discriminators):
            layers.append(ConvNet(DISC_IN_SIZE,
                                  DISC_OUT_SIZE,
                                  DISC_CAPACITY,
                                  DISC_N_LAYERS,
                                  DISC_KERNEL_SIZE,
                                  DISC_STRIDE))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.layers:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features
    
def split_features(self, features):
    feature_real = []
    feature_fake = []
    for scale in features:
        true, fake = zip(*map(
            lambda x: torch.split(x, x.shape[0] // 2, 0),
            scale,
        ))
        feature_real.append(true)
        feature_fake.append(fake)
    return feature_real, feature_fake

class ConvNet(nn.Module):

    def __init__(self, in_size, out_size, capacity, n_layers, kernel_size,
                 stride) -> None:
        super().__init__()
        channels = [in_size]
        channels += list(capacity * 2**np.arange(n_layers))

        if isinstance(stride, int):
            stride = n_layers * [stride]

        net = []
        for i in range(n_layers):
            if not isinstance(kernel_size, int):
                pad = (cc.get_padding(kernel_size[0],
                                      stride[i],
                                      mode="centered")[0], 0)
                s = (stride[i], 1)
            else:
                pad = cc.get_padding(kernel_size, stride[i],
                                     mode="centered")[0]
                s = stride[i]
            net.append(
                normalization(
                    nn.Conv1d(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=s,
                        padding=pad,
                    ), DISC_NORM_MODE))
            net.append(nn.LeakyReLU(.2))
        net.append(nn.Conv1d(channels[-1], out_size, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(x)
        return features

def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')
    
def mean_difference(target: torch.Tensor,
                    value: torch.Tensor,
                    norm: str = 'L1',
                    relative: bool = False):
    diff = target - value
    if norm == 'L1':
        diff = diff.abs().mean()
        if relative:
            diff = diff / target.abs().mean()
        return diff
    elif norm == 'L2':
        diff = (diff * diff).mean()
        if relative:
            diff = diff / (target * target).mean()
        return diff
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')
    
def hinge_gan(score_real, score_fake):
    loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
    loss_dis = loss_dis.mean()
    loss_gen = -score_fake.mean()
    return loss_dis, loss_gen