import pickle
from module import *
from funcs import *
from config import *
import torch
import torchaudio
import torchaudio.transforms as transforms

# Scope

if __name__ == '__main__':
    for name in ['PNO','STR','WND','GTR']:
        for i in range(1,9):
            waveform, sample_rate = torchaudio.load(f'/workspace/wss/instrument_set/{name}{i}.wav')
            data = waveform[0,:(96000//1024)*1024].view(-1,1024).to(DEVICE)
            data, _, _ = play_preprocess(data, 48000, 1, 'crepe')
            # 원본 오디오 파일 불러오기
            for j, x in enumerate(data):
                # 행을 오디오 데이터로 변환
                x = x.unsqueeze(0)  # 텐서를 [1, 1024]로 변경
                # 오디오 데이터를 WAV 파일로 저장
                filename = f"/workspace/wss/internal_wavetable/{name}_{str(10000+(i-1)*93+j+1)[1:]}.wav"
                torchaudio.save(filename, x.to('cpu'), 48000)