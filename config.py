import os
import torch

from models.convlstm import ConvLSTM_Model
from models.tau import TAU_Model
from models.ufno import UFNO_Model

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def set_model_params(args, in_channels=7, out_channels=4):
    if args.model_name == 'convlstm':
        model_params = {
            "in_steps": args.in_steps,
            "out_steps": args.out_steps,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "lr": args.lr,
            "nf": 64,
        }

        model_obj = ConvLSTM_Model

    elif args.model_name == 'tau':
        model_params = {
            "in_steps": args.in_steps,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "lr": args.lr,
            "hid_S": 64,
            "hid_T": 256,
            "N_S": 4,
            "N_T": 8,
            "spatio_kernel_enc": 3,
            "spatio_kernel_dec": 3,
            "alpha": 0.1,
            "tau": 0.1,
        }

        model_obj = TAU_Model

    elif args.model_name == 'ufno':
        modes = (args.in_steps // 2) + 1
        model_params = {
            "in_steps": args.in_steps,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "modes1": modes,
            "modes2": modes,
            "modes3": modes,
            "width": 32,
            "padding": False,
        }

        model_obj = UFNO_Model

    return model_params, model_obj


def set_trainer_params(args):
    ckpt_prefix = args.ckpt_prefix + '_{epoch:02d}'
    monitored_loss = "Validation Loss" # For all callbacks

    ckpt_params = {
        'monitor': monitored_loss,
        'filename': ckpt_prefix,
        'save_top_k': 3,
        'every_n_epochs': 1,
    }

    early_stop_params = {
        'monitor': monitored_loss,
        'patience': 20,
        'verbose': False,
        'mode': 'min',
    }

    ckpt_callback = ModelCheckpoint(**ckpt_params)
    early_stop_callback = EarlyStopping(**early_stop_params)

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=os.path.join(
            args.model_name, f"level_{len(args.model_list)}"),
    )

    trainer_params = {
        "max_epochs": args.epochs,
        "devices": [args.gpu] if torch.cuda.is_available() else 0,
        "enable_progress_bar": True,
        "deterministic": True,
        "callbacks": [ckpt_callback, early_stop_callback],
        "check_val_every_n_epoch": 1,
        "logger": logger,
    }

    return trainer_params
