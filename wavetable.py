from module import *
from funcs import *
from config import *
from collections import deque as dq
import matplotlib.pyplot as plt
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

    out = torch.empty(0)
    C, Q = 17, 5
    print(f'Query:{WAVEFORM_NAMES[Q]}, Condition:{WAVEFORM_NAMES[C]}')
    w_dim1 = C*2
    w_dim2 = C*2+1
    length = 1
    R = 4.5
    rotation = 1
    x_path = f'/workspace/wss/SerumDataset/{WAVEFORM_NAMES[Q]}_126.wav'
    w_1 = np.ones(1)*4.5 #R*np.cos((np.linspace(0,1,length)*2*np.pi*rotation))+4.5
    w_2 = np.one(1)*4.5 #R*np.sin((np.linspace(0,1,length)*2*np.pi*rotation))+4.5

    
    with torch.no_grad():
        i_tensor = torch.tensor(1j, dtype=torch.complex64)
        indices = [index for index, value in enumerate(db.file_list) if value == x_path]
        i = indices[0]
        datum = list(db[i])
        for j in range(5):
            if isinstance(datum[j], int):
                datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
            else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)
        _, _, _, _, mu_z, logvar_z, mu_w, logvar_w = wavespace(tuple(datum))
        w = mu_w
        print(f'w: {w}')
        for w0, w1 in zip(w_1, w_2):
            z = torch.zeros(1, Z_DIM).to(DEVICE) #z = (0,0,...,0)
            w[0, w_dim1] = w0
            w[0, w_dim2] = w1
            x_hat, _ = wavespace.decoder(w, z)
            x_hat = torch.exp(-x_hat)
            x_hat = torch.squeeze(x_hat/torch.max(x_hat)).to('cpu')
            x_hat = torch.concatenate((torch.zeros(1),x_hat))
            wave = ifft(i_tensor*x_hat)
            out = torch.concatenate((out, wave), dim=-1)
            
    plt.plot(out[:128])
    plt.savefig(f'/workspace/wss/fig/{C}-{Q}')
    path = f'/workspace/wss/generated_samples/{C}-{Q}.wav'#circular_c={c}-2R={int(2*R)}-rot={rotation}.wav'
    torchaudio.save(path, out.unsqueeze(0), SR)
    print(f'wav file saved:{path}')

    '''
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved'''