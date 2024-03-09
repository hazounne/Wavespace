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
    if CKPT_LOAD:
        wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
    else:
        wavespace = Wavespace().to(DEVICE)
    print(wavespace)

# Train.
    trainer = pl.Trainer(max_epochs=EPOCH,
                         resume_from_checkpoint=CKPT_TEST if CKPT_LOAD else None,
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