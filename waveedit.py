import os
import torchaudio
import torch.nn.functional as F

# in_dir containing the waveforms
cond = 'softwaves'
in_dir  = f'/workspace/wss/WaveEditDataset/{cond}'
out_dir = f'/workspace/wss/WaveEditDataset/processed_upscaled/{cond}'

# Scale factor for upsampling
scale_factor = 4

# Iterate over all files in the in_dir
for j, filename in enumerate(os.listdir(in_dir)):
    if filename.endswith('.WAV'):
        # Load the waveform
        filepath = os.path.join(in_dir, filename)
        waveform, sample_rate = torchaudio.load(filepath)
        
        # Upsample the waveform
        upsampled_waveform = F.interpolate(waveform.unsqueeze(0), scale_factor=scale_factor, mode='linear', align_corners=False).squeeze(0)
        
        # Save the upsampled waveform
        reshaped_waveform = upsampled_waveform.view(64,1024)
        for i in range(64):
            save_path =f'{out_dir}/{cond}_{str(10001+i+64*j)[1:]}.wav'
            torchaudio.save(save_path, reshaped_waveform[i, :].unsqueeze(0), 48000)
