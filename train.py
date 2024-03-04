import sys
from module import *
from funcs import *
from config import *
import os

torch.manual_seed(0)
torch.cuda.empty_cache()

train_loaders, test_loaders, val_loaders = data_build(
    DATASETS,
    [9], #1:train 0:test -1:valid, X:pass, else:n-fold
    BS=BS,
    loaderonly=True
    )

if __name__ == '__main__':
    wavespace = Wavespace().to(DEVICE)
    print(wavespace)
# Load Checkpoint.
    if CKPT_LOAD:
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

# Train.
    trainer = pl.Trainer(max_epochs=EPOCH,
                         accelerator='gpu',
                         devices=[GPU_NUM],
                         enable_progress_bar=True,
                         default_root_dir=PARENT_PATH / f"wss/log",
                         )
    trainer.fit(wavespace,
                train_dataloaders=train_loaders[0],
                )

##### Save ckpt
    # ckpt = {
    #     'state_dict': wavespace.state_dict(),
    #    'optimizer_state_dict': OPTIMIZER.state_dict()
    # }
    # torch.save(ckpt, CKPT_PATH)
    trainer.save_checkpoint(CKPT_PATH)
    print(f'CKPT saved to {CKPT_PATH}')

##### Test
    #trainer.test(wavespace, test_loaders[0])
    wandb.finish()