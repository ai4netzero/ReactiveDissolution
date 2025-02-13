from models.convlstm import ConvLSTM_Model
from models.tau import TAU_Model
from models.ufno import UFNO_Model


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