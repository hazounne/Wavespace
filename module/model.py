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
import time
# from torchviz import make_dot
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
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if LEARN_PRIORS:
            initial_mu_z = torch.tensor(MU_Z).to(DEVICE)
            self.mu_z_polar = nn.Parameter(torch.rand((2*len(WAVEFORMS),2*len(WAVEFORMS)-1))-0.5).mul(2*torch.pi).to(DEVICE).clone().detach() #the polar expression in -pi to pi given fixed r.
            self.mu_z_polar.requires_grad = True
            params.extend([self.mu_z_polar])
        else:
            initial_mu_z = torch.tensor(MU_Z).to(torch.float32).to(DEVICE)
            self.mu_z = initial_mu_z
        initial_logvar_z = torch.tensor(LOGVAR_Z).to(torch.float32).to(DEVICE)
        self.logvar_z = initial_logvar_z
        self.optim = torch.optim.Adam(params, LR, (0.9, 0.999))
        if LOSS_SCHEDULE:
            self.gen_opt_scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=1.0, end_factor=0.5, total_iters=1500)

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

    def polar_to_cartesian(self, angles):
        radius = 5
        num_of_angles = angles.shape[1]
        # Compute Cartesian coordinates
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        cartesian_coords = torch.zeros(angles.shape[0], num_of_angles + 1)  # Initialize Cartesian coordinates
    
        # Compute Cartesian coordinates for each dimension
        for i in range(num_of_angles):
            cartesian_coords[:, i] = radius * cos_angles[:, i]
            radius = radius * sin_angles[:, i]
        cartesian_coords[:, -1] = radius  # Last coordinate
        return cartesian_coords.to(DEVICE)

    def forward(self, batches, gen=False, return_decoder_time=False):
        #Encode
        if DATASET_TYPE == 'WAVETABLE':
            x, y = batches
        elif DATASET_TYPE == 'PLAY':
            x, y = batches
            x, f0, amp = play_preprocess(x, f_s=16000, n=X_DIM, f_0='crepe')
            x = x/amp
        mu_w, logvar_w, x, x_spec = self.encoder(x)

        if LEARN_PRIORS: self.mu_z = self.polar_to_cartesian(self.mu_z_polar)

        w = self.sampling(mu_w, logvar_w, y)

        #Decode
        if AB_D == 1:
            sc = get_semantic_conditions(x)
            w_sc = torch.cat((w, sc), dim=-1)
        else: w_sc = w
        start = time.time()
        x_hat, x_hat_spec = self.decoder(w_sc)
        end = time.time()
        decoder_time = end-start
        if wandb.run != None:
            wandb.log({
            'W': w,
            'xhat0': x_hat[0,:],
            'xhatvar': (torch.std(x_hat,dim=0)),
            })

        if gen: return x_hat
        elif return_decoder_time: return (x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y), decoder_time
        else: return x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y
    
    def training_step(self, batches, batch_idx):
        x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y = self(batches)
        
        # Calculate loss
        LOSS = self.loss_function(x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y, 'train')
        
        # Backpropagation
        self.optim.zero_grad()
        LOSS.backward(retain_graph=True)
        if LEARN_PRIORS:
            if self.mu_z_polar.grad is not None:
                self.mu_z_polar.grad = self.mu_z_polar.grad * PRIOR_COEF
        
        # Optimization step
        self.optim.step()
        
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
        SEMANTIC_LOSS_BATCH = (get_semantic_conditions(x) - get_semantic_conditions(x_hat)).abs() % 2*torch.pi # % only becomes to treats phase loss.
        PHASE_LOSS_BATCH = torch.minimum(2 * torch.pi - SEMANTIC_LOSS_BATCH[:, 4], SEMANTIC_LOSS_BATCH[:, 4]) / BS
        NOISE_LOSS_BATCH = SEMANTIC_LOSS_BATCH[:, 3] / BS
        SEMANTIC_LOSS_BATCH = torch.sum(SEMANTIC_LOSS_BATCH[:, :3], dim=-1) / BS
        KL_LOSS = torch.sum(self.KL(
                    mu_w,
                    logvar_w,
                    self.mu_z[y],
                    self.logvar_z[y],).unsqueeze(1), -1)


        SPECTRAL_LOSS = torch.sum(SPECTRAL_LOSS_BATCH)
        WAVEFORM_LOSS = torch.sum(WAVEFORM_LOSS_BATCH)  
        WAVEFORM_LOSS_COEF_MULTIPLIED = WAVEFORM_LOSS_COEF * (1 + exp(-self.current_epoch * WAVEFORM_LOSS_DECREASE_RATE) * (WAVEFORM_LOSS_MULTIPLIER - 1))      
        SEMANTIC_LOSS = torch.sum(SEMANTIC_LOSS_BATCH)
        PHASE_LOSS = torch.sum(PHASE_LOSS_BATCH)
        NOISE_LOSS = torch.sum(NOISE_LOSS_BATCH)
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

    def loss_values(self, x, x_hat, x_spec, x_hat_spec, mu_w, logvar_w, y):

        SPECTRAL_LOSS_BATCH = torch.sum((x_spec - x_hat_spec).pow(2), -1)
        WAVEFORM_LOSS_BATCH = torch.sum(torch.abs(x - x_hat), -1)
        SEMANTIC_LOSS_BATCH = (get_semantic_conditions(x) - get_semantic_conditions(x_hat)).abs() % 2*torch.pi # % only becomes to treats phase loss.
        BRIGHTNESS_LOSS_BATCH = torch.sum(SEMANTIC_LOSS_BATCH[:, 0], dim=-1)
        RICHNESS_LOSS_BATCH = torch.sum(SEMANTIC_LOSS_BATCH[:, 1], dim=-1)
        FULLNESS_LOSS_BATCH = torch.sum(SEMANTIC_LOSS_BATCH[:, 2], dim=-1)
        NOISE_LOSS_BATCH = SEMANTIC_LOSS_BATCH[:, 3]
        PHASE_LOSS_BATCH = torch.minimum(2 * torch.pi - SEMANTIC_LOSS_BATCH[:, 4], SEMANTIC_LOSS_BATCH[:, 4])
        KL_LOSS = torch.sum(self.KL(
                    mu_w,
                    logvar_w,
                    self.mu_z[y],
                    self.logvar_z[y],).unsqueeze(1), -1)


        SPECTRAL_LOSS = torch.sum(SPECTRAL_LOSS_BATCH).div(512).item()
        WAVEFORM_LOSS = torch.sum(WAVEFORM_LOSS_BATCH).div(512).item()
        BRIGHTNESS_LOSS = torch.sum(BRIGHTNESS_LOSS_BATCH).item()
        RICHNESS_LOSS = torch.sum(RICHNESS_LOSS_BATCH).item()
        FULLNESS_LOSS = torch.sum(FULLNESS_LOSS_BATCH).item()
        NOISE_LOSS = torch.sum(NOISE_LOSS_BATCH).item()
        PHASE_LOSS = torch.sum(PHASE_LOSS_BATCH).item()
        KL = torch.sum(KL_LOSS).item()
        return torch.tensor([WAVEFORM_LOSS, SPECTRAL_LOSS, BRIGHTNESS_LOSS, RICHNESS_LOSS, FULLNESS_LOSS, NOISE_LOSS, PHASE_LOSS, KL]).to(DEVICE)

    def configure_optimizers(self):
        pass