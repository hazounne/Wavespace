from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import deque as dq
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

    X = dq([])
    H = dq([])
    W = dq([])
    Y = dq([])
    N = db.__len__()

    #print(db.__len__())
    for i in range(N):
        if i%100==0: print(f'{i}/{N} done.')
        datum = list(db[i])
        for j in range(len(datum)):
            if isinstance(datum[j], int):
                if j==1: Y.append(datum[j])
                datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
            else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)
        _, x_hat, mu_w, logvar_w, _ = wavespace(tuple(datum))
        
        if DATASET_TYPE=='PLAY': _, x = overtone(datum[0], f_s=16000, n=N_OVERTONES, f_0=datum[3])
        elif DATASET_TYPE == 'WAVETABLE': x = datum[0]

        w = wavespace.sampling(mu_w, logvar_w)
        X.append(x.squeeze().to('cpu').tolist())
        W.append(w.squeeze().to('cpu').tolist())
        H.append(x_hat.squeeze().to('cpu').tolist())

    tsne = TSNE(n_components=2,
                  random_state=42,
                  learning_rate='auto',
                  init='random',
                  perplexity=300,
                )
    X_tsne = tsne.fit_transform(np.array(list(X)))
    W_tsne = tsne.fit_transform(np.array(list(W)))
    H_tsne = tsne.fit_transform(np.array(list(H)))
    # W_tsne = tsne.fit_transform(np.array(list(W)))

    # Separate data points by categories
    categories = np.unique(np.array(Y))

    # Create a color map for the categories
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(categories)))

    W = W_tsne
    # Create a scatter plot
    for i, category in enumerate(categories):
        plt.scatter(W[Y == category, 0],
                    W[Y == category, 1],
                    color=colors[i],
                    label=WAVEFORM_NAMES[category],
                    s=0.5)

    # Add labels and a title
    plt.legend(loc='upper right', title='conditions', fontsize='small')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('W')
    plt.savefig( PARENT_PATH / f'wss/fig/W_{CKPT_NAME}.png')
    print('W done')
    plt.clf()

    # Create a scatter plot
    for i, category in enumerate(categories):
        plt.scatter(X_tsne[Y == category, 0],
                    X_tsne[Y == category, 1],
                    color=colors[i],
                    label=WAVEFORM_NAMES[category],
                    s=0.5
                    )

    # Add labels and a title
    plt.legend(loc='upper right', title='conditions', fontsize='small')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('X')
    plt.savefig(PARENT_PATH / f'wss/fig/X_{CKPT_NAME}.png')
    print('X done')
    plt.clf()

#XHAT
    # Create a scatter plot
    for i, category in enumerate(categories):
        plt.scatter(H_tsne[Y == category, 0],
                    H_tsne[Y == category, 1],
                    color=colors[i],
                    label=WAVEFORM_NAMES[category],
                    s=0.5
                    )

    # Add labels and a title
    plt.legend(loc='upper right', title='conditions', fontsize='small')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('X_hat')
    plt.savefig(PARENT_PATH / f'wss/fig/H_{CKPT_NAME}.png')
    plt.clf()