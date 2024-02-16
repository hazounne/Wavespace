from module import Encoder, Decoder
import torch 

encoder = Encoder()
decoder = Decoder()

mu, log_var = encoder(torch.zeros(5, 1024))

def sampling(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)

w = sampling(mu, log_var)
print(decoder(w).size())


import numpy as np

def sum_sine_waves(amps, phases, frequencies, time_period=1, num_samples=512):

    time = np.linspace(0, time_period, num_samples)
    amp = amp / np.sum(amp)
    sum_wave = np.zeros_like(time)
    for amp, phase, freq in zip(amps, phases, frequencies):
        sum_wave += amp * np.sin(2 * torch.pi * freq * time + phase)
    return sum_wave

#print(sum_sine_waves([0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 3, 4]))