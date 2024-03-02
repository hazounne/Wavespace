from module import *
from funcs import *
from config import *
from collections import deque as dq
import matplotlib.pyplot as plt
import torchaudio

if __name__ == '__main__':
    wavespace = Wavespace().to(DEVICE)
    ###CHECKPOINT LOAD###
    load_ckpt = torch.load(CKPT_TEST)
    loaded_model_state_dict = load_ckpt['state_dict']
    if STAGE == 1:
        new_state_dict = wavespace.state_dict()
        for key in new_state_dict.keys():
            if not 'discriminator' in key:
                new_state_dict[key] = loaded_model_state_dict[key]
        wavespace.load_state_dict(new_state_dict)
    elif STAGE == 2: wavespace.load_state_dict(loaded_model_state_dict)

    print(f"checkpoint_loaded:{CKPT_TEST}")
    if STAGE == 2:
        for param in wavespace.encoder.parameters():
            param.requires_grad = False
    wavespace.eval()
    
    audio_dir = '/data3/NSynth/nsynth-test/audio/keyboard_acoustic_004-056-127.wav'
    AUDIO_SOURCE, _ = torchaudio.load(audio_dir)
    SIZE = (AUDIO_SOURCE.shape[-1]//1024)*1024
    with torch.no_grad():
        x = AUDIO_SOURCE[:,:SIZE].to(DEVICE)
        x = x.reshape(-1,1024)
        w = wavespace.encoder(x)[0]
        sc = dco_extractFeatures(x).to(DEVICE)
        x_hat = wavespace.decoder(torch.concatenate((w,sc),dim=-1))
        x_hat = x_hat.reshape(1,SIZE)
    name_pos=audio_dir.rfind("/")
    path = f'/workspace/wss/generated_samples{audio_dir[name_pos:]}'
    torchaudio.save(path, x_hat.cpu(), 16000)
    print(f'wav file saved:{path}')