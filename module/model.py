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
from math import exp
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
        gen_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if LEARN_PRIORS:
            initial_mu_z = torch.tensor(MU_Z).to(DEVICE)
            initial_logvar_z = torch.tensor(LOGVAR_Z).to(DEVICE)
            if PRIORS_RANDOM_INITIALISE: #if priors are randomly initialised by randn
                self.mu_z = nn.Parameter(torch.randn_like(initial_mu_z).to(DEVICE).clone().detach())
                self.logvar_z = nn.Parameter(torch.randn_like(initial_logvar_z).to(DEVICE).clone().detach())
            else:
                self.mu_z = nn.Parameter(initial_mu_z.clone().detach())
                self.logvar_z = nn.Parameter(initial_logvar_z.clone().detach())
            gen_params.extend([self.mu_z, self.logvar_z])
        else:
            initial_mu_z = torch.tensor(MU_Z).to(torch.float32).to(DEVICE)
            initial_logvar_z = torch.tensor(LOGVAR_Z).to(torch.float32).to(DEVICE)
            self.mu_z = initial_mu_z
            self.logvar_z = initial_logvar_z
        self.gen_opt = torch.optim.Adam(gen_params, 1e-3, (.5, .9))
        if LOSS_SCHEDULE:
            self.gen_opt_scheduler = torch.optim.lr_scheduler.LinearLR(self.gen_opt, start_factor=1.0, end_factor=0.1, total_iters=1500)
        #MISC
        self.i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)

    def sampling(self, mu, log_var, y):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std).mul(torch.exp(0.5*self.logvar_z[y])).add(self.mu_z[y])
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
            x, y = batches
        elif DATASET_TYPE == 'PLAY':
            x, y = batches
            x, f0, amp = play_preprocess(x, f_s=16000, n=X_DIM, f_0='crepe')
            x = x/amp
        mu_w, logvar_w, x, x_spec = self.encoder(x)
        w = self.sampling(mu_w, logvar_w, y)

        #Decode
        sc = get_semantic_conditions(x)
        w_sc = torch.cat((w, sc), dim=-1)
        x_hat, x_hat_spec = self.decoder(w_sc)
        if wandb.run != None:
            wandb.log({
            'W': w,
            'xhat0': x_hat[0,:],
            'xhatvar': (torch.std(x_hat,dim=0)),
            })

        if gen: return x_hat
        else: return x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y
    
    def training_step(self, batches, batch_idx):
        x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y = self(batches)
        
        # Calculate loss
        LOSS = self.loss_function(x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y, 'train')
        
        # Backpropagation
        self.gen_opt.zero_grad()
        LOSS.backward(retain_graph=True)
        
        # Apply prior coefficient to gradients
        if LEARN_PRIORS:
            if self.mu_z.grad is not None:
                self.mu_z.grad *= PRIOR_COEF
            if self.logvar_z.grad is not None:
                self.logvar_z.grad *= PRIOR_COEF
            
            # Detach gradients
            self.mu_z.grad.detach_()
            self.logvar_z.grad.detach_()
        
        # Optimization step
        self.gen_opt.step()
        
        # Optionally adjust learning rate
        if LOSS_SCHEDULE:
            self.gen_opt_scheduler.step()
        
        return LOSS


    def gen(self, x):
        x_hat = self(x, gen=True)
        return x_hat
    
    def loss_function(self, x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y,
                      process='train'):

        SPECTRAL_LOSS_BATCH = torch.sum((x_spec - x_hat_spec).pow(2), -1)/BS
        WAVEFORM_LOSS_BATCH = torch.sum(torch.abs(x - x_hat), -1)/BS
        SEMANTIC_LOSS_BATCH = (get_semantic_conditions(x) - get_semantic_conditions(x_hat)).abs() % 2*torch.pi
        PHASE_LOSS_BATCH = torch.minimum(2 * torch.pi - SEMANTIC_LOSS_BATCH[:, 4], SEMANTIC_LOSS_BATCH[:, 4]) / BS
        NOISE_LOSS_BATCH = SEMANTIC_LOSS_BATCH[:, 3] / BS
        SEMANTIC_LOSS_BATCH = torch.sum(SEMANTIC_LOSS_BATCH[:, :3], dim=-1) / BS
        KL_LOSS = torch.sum(self.KL(
                    mu_w,
                    logvar_w,
                    self.mu_z[y],
                    self.logvar_z[y],).unsqueeze(1), -1)
        FEATURE_MATCHING_LOSS = 0
        ADVERSARIAL_LOSS = 0

        SPECTRAL_LOSS = torch.sum(SPECTRAL_LOSS_BATCH)
        WAVEFORM_LOSS = torch.sum(WAVEFORM_LOSS_BATCH)  
        WAVEFORM_LOSS_COEF_MULTIPLIED = WAVEFORM_LOSS_COEF * (1 + exp(-self.current_epoch * WAVEFORM_LOSS_DECREASE_RATE) * (WAVEFORM_LOSS_MULTIPLIER - 1))      
        SEMANTIC_LOSS = torch.sum(SEMANTIC_LOSS_BATCH)
        PHASE_LOSS = torch.sum(PHASE_LOSS_BATCH)
        NOISE_LOSS = torch.sum(NOISE_LOSS_BATCH)
        #SPECTRAL_LOSS_COEF_COMPENSATION = WAVEFORM_LOSS_MULTIPLIER*WAVEFORM_LOSS_COEF - WAVEFORM_LOSS_COEF_MULTIPLIED
        LOSS = (SPECTRAL_LOSS_COEF * SPECTRAL_LOSS_BATCH
                + WAVEFORM_LOSS_COEF_MULTIPLIED * WAVEFORM_LOSS_BATCH
                + SEMANTIC_LOSS_COEF * SEMANTIC_LOSS_BATCH
                + PHASE_LOSS_COEF * PHASE_LOSS_BATCH
                + NOISE_LOSS_COEF * NOISE_LOSS_BATCH
                + KL_LOSS_COEF * KL_LOSS).sum()
        KL = torch.sum(KL_LOSS)
        if wandb.run != None:
            wandb.log({f'SPECTRAL_LOSS_BATCH': SPECTRAL_LOSS_BATCH,
                       f'WAVEFORM_LOSS_BATCH': WAVEFORM_LOSS_BATCH,
                       f'SEMANTIC_LOSS_BATCH': SEMANTIC_LOSS_BATCH,
                       f'SPECTRAL_LOSS': SPECTRAL_LOSS,
                       f'WAVEFORM_LOSS': WAVEFORM_LOSS,
                       f'SEMANTIC_LOSS': SEMANTIC_LOSS,
                       f'PHASE_LOSS': PHASE_LOSS,
                       f'NOISE_LOSS': NOISE_LOSS,
                       f'RECONSTRUCTION_LOSS': SPECTRAL_LOSS + WAVEFORM_LOSS,
                       f'KL_LOSS': KL_LOSS,
                       f'{process}_LOSS': SPECTRAL_LOSS + WAVEFORM_LOSS + SEMANTIC_LOSS + PHASE_LOSS + NOISE_LOSS,
                       f'WAVEFORM_LOSS_COEF_MULTIPLIED': WAVEFORM_LOSS_COEF_MULTIPLIED,
                       })
        assert not (torch.isnan(LOSS).any())
        return LOSS

    def configure_optimizers(self):
        pass