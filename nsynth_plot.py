from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ##LOAD
    load_ckpt = torch.load(CKPT_TEST)
    loaded_model_state_dict = load_ckpt['state_dict']
    # loaded_optimizer_state_dict = load_ckpt['optimizer_state_dict']

    wavespace = Wavespace()
    wavespace.load_state_dict(loaded_model_state_dict)
    wavespace = wavespace.to(DEVICE) #after train/test, the model automatically set to CPU
    wavespace.eval()
    #optimizer = optim.Adam(wavespace.parameters(), lr=0.001)
    #optimizer.load_state_dict(loaded_optimizer_state_dict)
    db = DatasetBuilder(file_list=DATASETS[0])

    # Plot.
    for C, Q in [(0,1), (1,1), (2,1)]:#range(N_CONDS):
        #C = 0
        r1=(0,9) #Range 1
        r2=(0,9) #Range 2
        num_x = 9 #Number of Plots 1
        num_y = 9 #Number of Plots 2
        w_1_test=np.linspace(r1[0],r1[1],num_x)
        w_2_test=np.linspace(r2[1],r2[0],num_y)

        query = 'bass_synthetic_033-048-127' #Condition
        print(f'Query:{query}, Condition:{WAVEFORM_NAMES[C]}')

        query_path = f'/data3/NSynth/nsynth-test/audio/{query}.wav'
        with torch.no_grad():
            i_tensor = torch.tensor(1j, dtype=torch.complex64)
            indices = [index for index, value in enumerate(db.file_list) if value == query_path]
            i = indices[0]
            datum = list(db[i])
            for j in range(5):
                if isinstance(datum[j], int):
                    datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
                else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)
            #datum[0] = torch.zeros(1,X_DIM).to(DEVICE)
            #for i in range(1,12):
            #    datum[0][0,i] = 1/2*i
            #datum[0][0,0] = 1/2**12
            _, _, _, mu_w, logvar_w, _ = wavespace(tuple(datum))
            #z = mu_z #= wavespace.sampling(mu_z, logvar_z)
            w = mu_w
            print(f'W: {w.tolist()}')
            fig, axes = plt.subplots(len(w_1_test), len(w_2_test), figsize=(80, 48))
            for m, w_2 in enumerate(w_2_test):
                for n, w_1 in enumerate(w_1_test):
                    #w = torch.zeros(1, W_DIM).to(DEVICE)
                    #w = torch.Tensor([])
                    for i in range(SUB_DIM):
                        w[0, i+C*SUB_DIM] = w_1
                    x_hat, _ = wavespace.decoder(w)
                    x_hat = torch.squeeze(x_hat/torch.max(x_hat)).to('cpu')
                # axes[m,n].plot(x_hat.to('cpu'), linewidth=1)
                    x_hat = torch.concatenate((torch.zeros(1),x_hat))
                    #dc_x_hat = torch.concatenate((torch.zeros(1), x_hat), dim=0)
                    axes[m,n].plot(ifft(i_tensor*x_hat), linewidth=6, c='black')
                    axes[m,n].set_title(f"w = {w_1}, {w_2}")
                    axes[m,n].set_xticks([])
                    axes[m,n].set_yticks([])
                    axes[m,n].grid(True)
            plt.tight_layout()
            folder_name = f'wss/fig/W/{CKPT_NAME}/C{WAVEFORM_NAMES[C]}/Q{query}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            directory = folder_name + f'/{r1}{r2}{num_x}x{num_y}.png'
            plt.savefig(directory)
            print(f'plot saved to {directory}')