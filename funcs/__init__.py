from .functions import *
from funcs import crepe

def play_preprocess(x: np.ndarray,
                          f_s: int,
                          n: int,
                          f_0: int) -> torch.Tensor:
  
  if f_0=='crepe':
    cr = crepe.CREPE("full").to(DEVICE)
    _, f_0, _, _ = cr.predict(x.flatten(), SR, batch_size=x.shape[0])

  # Apply FFT to calculate the DFT
  spectrum = torch.fft.rfft(x)
  #freqs = torch.fft.fftfreq(len(spectrum), 1/f_s) obsolete
  magnitude_spectrum = torch.abs(spectrum)
  magnitude_spectrum = torch.concatenate((magnitude_spectrum, torch.zeros(x.shape[0],1).to(DEVICE)), dim=-1) #if bigger 0
  # Find harmonic peaks
  f0_index = torch.round(f_0 * RAW_LEN / f_s).to(DEVICE)
  #harmonic_indices = [int(f0_index * h) for h in range(1, n+1)]
  harmonic_indices = torch.arange(1, X_DIM+1).to(DEVICE) * f0_index.view(-1, 1)
  harmonic_indices = harmonic_indices.clamp(max=RAW_LEN//2+1) # indices no bigger than magnitude_spectrum.size(1). if bigger 0.
  harmonic_magnitudes = magnitude_spectrum.gather(1, harmonic_indices.to(torch.int64))
  # magnitude_spectrum = torch.concatenate((magnitude_spectrum,
  #                                       torch.zeros(max(0, int(f0_index * n)
  #                                       -len(magnitude_spectrum) + 1))
  #                                       )
  #     )
  # harmonic_magnitudes = magnitude_spectrum[harmonic_indices]
  amp = torch.sum(harmonic_magnitudes.pow(2), dim=-1).sqrt().unsqueeze(-1)
  harmonic_magnitudes /= amp #energy normaliser
  return harmonic_magnitudes, f_0, amp