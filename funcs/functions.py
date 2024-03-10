# quantization
from config import *
import random
import torch.fft as fft
import random
import torch
import matplotlib.pyplot as plt
from funcs import crepe

def harmonic_structure(x):
  fft_results = fft(x)
  amp = torch.sqrt(torch.abs(fft_results)).float()
  print(amp.shape)
  if torch.sum(amp, dim=0) < 0.2: amp = torch.zeros_like(amp)
  #if torch.isnan(torch.sum(amp/torch.sum(amp))): return amp
  return amp/torch.sum(amp, dim=0) #normalization

def plot_save(x,name: str, acc=False) -> None:
  plt.plot(x)
  plt.savefig(PARENT_PATH/ f'wss/fig/{name}')
  if not acc: plt.show()

def log(x):
  return torch.log(torch.max(x+EPSILON,torch.ones_like(x)*EPSILON))

def idft(x:torch.tensor) -> torch.tensor:
    return fft.irfft(torch.tensor(1j, dtype=torch.complex64)*x)
def get_padding(k, s=1, d=1, mode="centered") -> tuple:
    """
    Computes 'same' padding given a kernel size, stride an dilation.

    Parameters
    ----------

    k: int
        k of the convolution

    stride: int
        stride of the convolution

    dilation: int
        dilation of the convolution

    mode: str
        either "centered", "causal" or "anticausal"
    """
    if k == 1: return (0, 0)
    p = (k - 1) * d + 1
    half_p = p // 2
    if mode == "centered":
        p_right = p // 2
        p_left = (p - 1) // 2
    elif mode == "causal":
        p_right = 0
        p_left = p // 2 + (p - 1) // 2
    elif mode == "anticausal":
        p_right = p // 2 + (p - 1) // 2
        p_left = 0
    else:
        raise Exception(f"Padding mode {mode} is not valid")
    return (p_left, p_right)

def get_semantic_conditions(x: torch.Tensor):
    bs = x.shape[0]
    waveform_length = x.shape[-1]
    N = waveform_length #* tile_num
    Nh = N // 2 + 1
    spec = fft.rfft(x)
    # calculate power spectrum
    spec_pow = torch.real(spec * torch.conj(spec) / N)

    total = torch.sum(spec_pow, -1, keepdim=True)

    centroid = torch.sum(spec_pow * torch.linspace(0, 1, Nh).to(DEVICE), -1, keepdim=True) / total
    
    k = torch.tensor([5.5]).to(DEVICE)
    brightness = log(centroid * (torch.exp(k) - 1) + 1) / k

    spread = torch.sqrt(torch.sum(spec_pow * (torch.linspace(0, 1, Nh).to(DEVICE).unsqueeze(0).tile((bs,1)) - centroid).pow(2), -1, keepdim=True) / total)
    k = torch.tensor([5.5]).to(DEVICE)
    richness = log(spread * (torch.exp(k) - 1) + 1) / k

    difference = torch.sum((torch.diff(x)).abs(), -1, keepdim=True) / (waveform_length-1)
    k = torch.tensor([5.5]).to(DEVICE)
    noisiness = log(difference * (torch.exp(k) - 1) + 1) / k
    noisiness = noisiness

    hnumber = int(waveform_length / 2) - 1
    odd_power = torch.sum(spec_pow[:, 1::2], -1, keepdim=True)
    fullness = 1 - odd_power / total

    # symmetry = torch.angle(spec.sum(-1, keepdim=True))
    symmetry = torch.sum((x[:,:512] + torch.flip(x[:, 512:], dims=[1])).abs(), -1, keepdim=True)
    brightness[total == 0] = -1
    richness[total == 0] = -1
    noisiness[total == 0] = -1
    fullness[total == 0] = -1
        
    return torch.concat((brightness, richness, fullness, noisiness, symmetry), dim=-1)

def play_preprocess(x: np.ndarray,
                          f_s: int,
                          n: int,
                          f_0: int) -> torch.Tensor:

    if f_0 == 'crepe':
        cr = crepe.CREPE("full").to(DEVICE)
        f_0 = cr.predict(x, SR, batch_size=x.shape[0])
    spectrum = torch.fft.rfft(x)
    f0_index = torch.round(f_0 * RAW_LEN / f_s).to(DEVICE)
    harmonic_indices = torch.arange(1, X_DIM+1).to(DEVICE) * f0_index.unsqueeze(1)
    harmonic_indices = harmonic_indices.clamp(max=RAW_LEN//2).to(torch.int64) # indices no bigger than half
    harmonic_spectrum = torch.gather(spectrum, 1, harmonic_indices)
    harmonic_spectrum = torch.cat((torch.zeros(spectrum.shape[0], 1).to(DEVICE), harmonic_spectrum), -1)
    harmonic_waveform = torch.fft.irfft(harmonic_spectrum)
    harmonic_amp = harmonic_spectrum.abs()
    amp = torch.sum(harmonic_amp.pow(2), dim=-1).div(x.shape[1] / 2).sqrt().unsqueeze(-1)
    #harmonic_magnitudes /= amp
    return harmonic_waveform, f_0, amp