from module import *
from config import *
if __name__ == '__main__':
    db = DatasetBuilder(file_list=DATASETS[0])
    filelist = ['bass_electronic_018-052-075',
                'bass_electronic_018-027-075',
                'bass_electronic_018-040-100',
                'flute_acoustic_002-079-127',
                'flute_acoustic_002-103-075',
                'flute_synthetic_000-045-127',
                'string_acoustic_080-026-025',
                'string_acoustic_056-050-127',
                'string_acoustic_014-062-127',]
    for filename in filelist:
        fullname = f'/data3/NSynth/nsynth-test/audio/{filename}.wav'
        i = db.file_list.index(fullname)
        datum = list(db[i])
        datum[0] = datum[0].reshape(1,-1).to(DEVICE)
        
        if DATASET_TYPE=='PLAY': _, x = overtone(datum[0], f_s=16000, n=N_OVERTONES, f_0='crepe')
        else: raise RuntimeError('set DATASET_TYPE to PLAY.')
        x = torch.concatenate((torch.zeros(1,1).to(DEVICE),x,torch.zeros(1,1025-65).to(DEVICE)), dim=-1)

        i_tensor = torch.tensor(1j).to(DEVICE)
        out = ifft(i_tensor*x).to('cpu')
        path = PARENT_PATH / f'wss/generated_samples/WF_{filename}.wav'
        torchaudio.save(path, out, 48000)