from config import *
from funcs import *
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as normalization
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as T

def concat(a,b):
    return torch.concatenate((a, b), dim=-1)

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        self._pad = kwargs.get("padding", (0, 0))
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs) # self._pad 값을 받고 에러 안뜨게 padding=0으로 만든 후 Conv1d 상속

    def forward(self, x):
        x = nn.functional.pad(x, self._pad)
        
        out = nn.functional.conv1d(
            x, 
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if not TRAINING:
            print(f'Conv {x.shape[1:]} -> {out.shape[1:]}') # console_check
        return out

class TrConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        self._pad = kwargs.get("padding", (0, 0))
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs) # self._pad 값을 받고 에러 안뜨게 padding=0으로 만든 후 Conv1d 상속

    def forward(self, x):
        x = nn.functional.pad(x, self._pad)
        out = torch.nn.functional.conv_transpose1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        if not TRAINING:
            print(f'TrConv {x.shape[1:]} -> {out.shape[1:]}') # console_check
        return out
    
# stride에 mod 2 (rave)
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        relu = nn.LeakyReLU(0.2)
        self.net = nn.Sequential(relu, TrConv1d(in_channels=in_channels, 
                                                out_channels = out_channels, 
                                                kernel_size = kernel_size, 
                                                stride = stride))

    def forward(self, x):
        out = self.net(x)
        # if not TRAINING:
        #     print(f'{x.size()} -> {out.size()}') # architectural_dimensional_fidelity_check
        return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        relu = nn.LeakyReLU(0.2)

        self.net = nn.Sequential(relu, nn.Conv1d(channels, channels, kernel_size=1, stride=1))

    def forward(self, x):
        return x + self.net(x)


##################################
# Defines Each part of the model.
##################################
'''
class Block(nn.Module):
    def __init__(
                self,
            ):
        super().__init__()

        net = []
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
'''
# class N_Encoder(nn.Module):
#     def __init__(
#                 self,
#             ):
#         super().__init__()

#         net = []
#         net.append(nn.Linear(X_DIM,1))
#         self.net = nn.Sequential(*net)

#     def forward(self, x):
#         return self.net(x)

class Encoder(nn.Module):
    def __init__(
                self,
            ):
        super().__init__()
        relu = nn.LeakyReLU(0.2)

        if BLOCK_STYLE == 'CONV1D':
            net = []
            for i in range(len(ENC_H)-1):
                net.extend([Conv1d(ENC_H[i],
                                   ENC_H[i+1],
                                   kernel_size=ENC_K[i],
                                   stride=ENC_S[i],
                                   padding=get_padding(ENC_K[i], ENC_S[i])),
                            relu, nn.BatchNorm1d(ENC_H[i+1])
                ])
            self.net = nn.Sequential(*net)
            self.linear = nn.Sequential(nn.Linear(ENC_H[-1], LATENT_LEN*2), relu)
        
        if BLOCK_STYLE == 'DDSP':
            self.mfcc = T.MFCC(
                sample_rate=SR,
                n_mfcc=13,
                melkwargs={'n_fft': 400, 'hop_length': 2049, 'n_mels': 23}
            )

            # Convolutional layer
            self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

            # GRU layer
            self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

            # Fully connected (Dense) layer
            self.fc = nn.Linear(64, W_DIM*2)

    def forward(self, x):
        if BLOCK_STYLE == 'CONV1D':
            x = self.net(x.unsqueeze(1))
            x = torch.squeeze(x, dim=2)
            w = self.linear(x)
            return w[:, :W_DIM], w[:, W_DIM:2*W_DIM]
        if BLOCK_STYLE == 'DDSP':
            # Input shape: (batch_size, 13, 1)
            x = self.mfcc(x)
            x = x.permute(0, 2, 1)  # Change to (batch_size, 1, 13) for Conv1D

            # Convolutional layer
            x = self.conv1d(x)

            # GRU layer
            _, x = self.gru(x.permute(0, 2, 1))  # Adjust dimensions for GRU

            # Fully connected layer
            w = self.fc(x[-1, :, :])  # Take the output from the last time step

            return w[:, :W_DIM], w[:, W_DIM:2*W_DIM]

class Decoder(nn.Module):
    def __init__(
                self,
            ):
        super().__init__()
        relu = nn.LeakyReLU(0.2)
        tanh = nn.Tanh()
        relu_out = nn.ReLU()

        if BLOCK_STYLE == 'CONV1D':
            self.linear = nn.Sequential(nn.Linear(LATENT_LEN, DEC_H[0]), relu)
            if DECODER_STYLE == 'SPECTRAL_COMBINED':
                self.output_dense = nn.Sequential(Conv1d(DEC_H[-1], 1, kernel_size=1, stride=1), tanh)
            elif DECODER_STYLE == 'SPECTRAL_SEPARATED':
                self.amp_dense = nn.Sequential(Conv1d(DEC_H[-1], 1, kernel_size=2, stride=2), relu) #we may take negative values to spin clockwise but we followed the practical way of additive synthesizers which only uses 0 and positive values.
                self.phase_dense = nn.Sequential(Conv1d(DEC_H[-1], 1, kernel_size=2, stride=2), tanh)
            net = []
            for i in range(len(DEC_H)-1):
                net.extend([UpSampleBlock(DEC_H[i], DEC_H[i+1], kernel_size=DEC_K[i], stride=DEC_S[i])])
                for j in range(RES_BLOCK_CONV_NUM):
                    net.extend([ResBlock(DEC_H[i+1])])
            self.net = nn.Sequential(*net)
        
        if BLOCK_STYLE == 'DDSP':
            self.mfcc = T.MFCC(
                sample_rate=SR,
                n_mfcc=13,
                melkwargs={'n_fft': 400, 'hop_length': 2049, 'n_mels': 23}
            )

            # Convolutional layer
            self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

            # GRU layer
            self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

            # Fully connected (Dense) layer
            self.fc = nn.Linear(64, W_DIM*2)

    def forward(self, x):
        if BLOCK_STYLE == 'CONV1D':
            x = self.linear(x).unsqueeze(2)
            x = self.net(x)
            if DECODER_STYLE == 'SPECTRAL_COMBINED':
                x = self.output_dense(x).squeeze()
                amp, phase = x[:, :512], x[:, 512:]
            elif DECODER_STYLE == 'SPECTRAL_SEPARATED':
                amp = self.amp_dense(x).squeeze()
                phase = self.phase_dense(x).squeeze()

            complex_expression = torch.concat((torch.zeros(x.shape[0],1).to(DEVICE), amp*torch.exp(1j * phase)), dim=-1)
            x_hat = idft(complex_expression)
            amp_overall = torch.sum(x_hat.pow(2), dim=-1).sqrt().unsqueeze(-1)
            x_hat = x_hat/amp_overall
            x_hat = x_hat*NORMALISED_ENERGY
            return x_hat

        if BLOCK_STYLE == 'DDSP':
            # Input shape: (batch_size, 13, 1)
            x = self.mfcc(x)
            x = x.permute(0, 2, 1)  # Change to (batch_size, 1, 13) for Conv1D

            # Convolutional layer
            x = self.conv1d(x)

            # GRU layer
            _, x = self.gru(x.permute(0, 2, 1))  # Adjust dimensions for GRU

            # Fully connected layer
            w = self.fc(x[-1, :, :])  # Take the output from the last time step

            return w

    #OBSOLÈTE
    def sum_sine_waves(self, amps, phases, frequencies, time_period=1, num_samples=1024):
        batch_size, _ = amps.size()
        time = torch.linspace(0, time_period, num_samples).repeat((batch_size, 1))
        sum_wave = torch.zeros((batch_size, num_samples))

        amps = torch.permute(amps, [1, 0]) # for문에서 배치 별로 봐야함
        phases = torch.permute(phases, [1, 0])
        for amp, phase, freq in zip(amps, phases, frequencies):
            phase = torch.permute(phase.repeat((1024, 1)), [1, 0]) # phase한꺼번에 계산하려하니...
            amp = torch.permute(amp.repeat((1024, 1)), [1, 0])

            sum_wave += amp * torch.sin(2 * torch.pi * freq * (time + phase))
        return sum_wave

class CONV1D(nn.Module):
    def __init__(
                self,
            ):
        super().__init__()

        self.unit = UNIT()
        self.rep = 3
    def forward(self, x):
        for _ in range(self.rep): x = self.unit(x)
        return x
    
class UNIT(nn.Module):
    def __init__(
                self,
            ):
        super().__init__()

        net = []
        net.extend([normalization(nn.Linear(W_DIM,W_DIM)),
                    nn.ReLU()])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)