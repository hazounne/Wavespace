from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt
import math

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
    for C, Q in [(3, 6), (3, 9), (4, 1), (2, 1), (5, 15), (5, 16), (8, 6), (10, 6)]:#range(N_CONDS):
        #C = 0
        r=(0,9) #Range 1
        theta=(0,np.pi/2) #Range 2
        num_x = 5 #Number of Plots 1
        num_y = 5 #Numbrer of Plots 2
        r_list=np.linspace(r[0],r[1],num_x)
        theta_list=np.linspace(theta[1],theta[0],num_y)

        query_name = WAVEFORM_NAMES[Q]
        query_num = '126'
        query = f'{query_name}_{query_num}' #Condition
        print(f'Query:{query}, Condition:{WAVEFORM_NAMES[C]}')

        query_path = PARENT_PATH / f'wss/SerumDataset/{query}.wav'
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
            print(f'W: {w}')
            fig, axes = plt.subplots(len(r_list), len(theta_list), figsize=(80, 48))
            for m, current_r in enumerate(r_list):
                for n, current_theta in enumerate(theta_list):
                    w[0, 0+C*2] = math.cos(current_theta)*current_r
                    w[0, 1+C*2] = math.sin(current_theta)*current_r
                    x_hat, _ = wavespace.decoder(w)
                    x_hat = torch.squeeze(x_hat/torch.max(x_hat)).to('cpu')
                # axes[m,n].plot(x_hat.to('cpu'), linewidth=1)
                    x_hat = torch.concatenate((torch.zeros(1),x_hat))
                    #dc_x_hat = torch.concatenate((torch.zeros(1), x_hat), dim=0)
                    axes[m,n].plot(ifft(i_tensor*x_hat), linewidth=6, c='black')
                    axes[m,n].set_title(f"w = {current_theta}, {current_r}")
                    axes[m,n].set_xticks([])
                    axes[m,n].set_yticks([])
                    axes[m,n].grid(True)
            plt.tight_layout()
            folder_name = f'wss/fig/Wc/{CKPT_NAME}/C{WAVEFORM_NAMES[C]}/Q{query}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            directory = folder_name + f'/{r}{theta}{num_x}x{num_y}.png'
            plt.savefig(directory)
            print(f'plot saved to {directory}')