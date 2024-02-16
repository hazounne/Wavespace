from config import *
from .blocks import *
from funcs import *

import matplotlib.pyplot as plt
from torch.fft import rfft as fft
from torch.fft import irfft as ifft

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch
import pytorch_lightning as pl
from itertools import chain

from time import time as T

# Wavespace: Explorable Wavetable Synthesizer
class Wavespace(pl.LightningModule):
    def __init__(
        self,
        encoder = Encoder(),
        decoder = Decoder(),
    ):
        super().__init__()
        
        #BLOCKS
        self.encoder = encoder
        self.decoder = decoder
        self.mu_w = torch.tensor(MU_W).to(DEVICE)
        self.logvar_w = torch.tensor(LOGVAR_W).to(DEVICE)
        #self.prior = torch.zeros((BS,W_DIM)).to(DEVICE) #have same prior
        #OPTIMIZERS
        self.optim_1 = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=LR)

        #LR SCHEDULERS
        if LOSS_SCHEDULE:
            milestones = [3 ** i for i in range(7)]
            gamma = 0.1 ** (1 / 7)
            self.scheduler1 = lr_scheduler.MultiStepLR(self.optim_1, milestones=milestones, gamma=gamma)

        #MISC
        self.midi_to_hz = torch.Tensor([440 * (2 ** ((midi_pitch - 69) / 12)) for midi_pitch in range(128)]).to(DEVICE)
        self.i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)
        self.k = 0.002  # Tunable parameter for the rate of increase

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def KL(self, mu1, logvar1, mu2, logvar2):
        std1 = torch.exp(0.5 * logvar1)
        std2 = torch.exp(0.5 * logvar2)
        return torch.sum(log(std2)
                        - log(std1)
                        + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2)
                        - 0.5, dim=-1)
        
    def forward(self, batches, gen=False):
        #Encode
        if DATASET_TYPE == 'WAVETABLE':
            x, y, pos, amp = batches
            pos = pos.reshape(-1,1)

        elif DATASET_TYPE == 'PLAY':
            x, f0, amp = play_preprocess(x, f_s=16000, n=X_DIM, f_0='crepe') #round(`midi_to_hz`(pitch)))
        mu_w, logvar_w = self.encoder(x)
        w = self.sampling(mu_w, logvar_w)

        #Decode
        if DATASET_TYPE == 'WAVETABLE': x_hat = self.decoder(w)
        elif DATASET_TYPE == 'PLAY': x_hat = self.decoder(w, amp, f0)
        if wandb.run != None:
            wandb.log({
            'W': w,
            'xhat0': x_hat[0,:],
            'xhatvar': (torch.std(x_hat,dim=0)),
            })

            # if self.global_step == 10:
            #     data = [[x, y] for (x, y)
            #             in zip(list(range(1024)), x_hat[0,:].cpu().detach().numpy().tolist())
            #             ]
            #     table = wandb.Table(data=data, columns=["x", "y"])
            #     wandb.log(
            #         {
            #             "my_custom_plot_id": wandb.plot.line(
            #                 table, "x", "y", title="Plot"
            #             )
            #         }
            #     )

        if gen: return x_hat
        else: return x, x_hat, mu_w, logvar_w, y
    
    def training_step(self, batches):
        x, x_hat, mu_w, logvar_w, y = self(batches)
        #loss
        loss = self.loss_function(x, x_hat, mu_w, logvar_w, y, 'train')
        #optimize
        self.optim_1.zero_grad()
        loss.backward(retain_graph=True)
        self.optim_1.step()
        if LOSS_SCHEDULE: self.scheduler1.step()
        return loss

    def gen(self, x):
        x_hat = self(x, gen=True)
        return x_hat

    def kl_annealing_schedule(self, epoch, max_epochs):
        #a sigmoid function
        return 2 / (1 + np.exp(-self.k * (epoch - max_epochs) / 2))
    
    def loss_function(self, x, x_hat,
                      mu_w, logvar_w, y,
                      process='train',
                    ):
        x_fft, x_hat_fft = torch.abs(fft(x)), torch.abs(fft(x_hat))
        spectral_difference = x_fft - x_hat_fft
        #wave_difference = x- x_hat
        L1_ms = torch.sum(spectral_difference.pow(2), -1)
        #L1_log = torch.sum(log(torch.abs(spectral_difference)), -1) #*BETA
        #L1_w = torch.sum(torch.abs(wave_difference), -1)
        L1 = L1_ms/BS #(L1_ms + L1_log)/BS

        L2 = torch.sum(self.KL(
                    mu_w,
                    logvar_w,
                    self.mu_w[y],
                    self.logvar_w[y],).unsqueeze(1), -1)

        loss = (B1 * L1 #RECON
                + B2 * L2 #KL
                ).sum()
        
        if wandb.run != None:
            wandb.log({f'L1': L1,
                       f'L2': L2,
                       f'{process}_Loss': loss,
                       })
        assert not (torch.isnan(loss).any())
        return loss

    def configure_optimizers(self):
        pass