import argparse
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from functools import partial

from data_loader import *
from preprocessing import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.multiprocessing import set_sharing_strategy

from config import *
from utils import *

def main(args):
    TRAIN_FILES = args.train_files
    VAL_FILES = args.val_files
    
    fix_randseeds(12345) 
    set_sharing_strategy("file_system")

    dataset_path = args.dataset_path
    dataset_files = sorted(os.listdir(dataset_path))

    train_files, val_files = train_test_split(
        dataset_files, train_size=TRAIN_FILES, test_size=VAL_FILES)

    data_list_train = read_full_dataset(
        train_files, dataset_path=dataset_path)
    
    data_list_val = read_full_dataset(
        val_files, dataset_path=dataset_path)
    
    dissolution_steps = 100
    input_scaler_list = estimate_input_scalers(
        data_list_train, scaler_type='std')
    outputs = args.outputs.split(",")

    preprocess_data_cube = partial(
        preprocess_data_cube_extended,
        outputs=outputs)
   
    dataset_train = DissolutionDataset(
        data_list_train,
        input_scaler_list,
        preprocess_data_cube_fnc=preprocess_data_cube,
        n_sample_files=TRAIN_FILES,
        dissolution_steps=dissolution_steps,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        extra_features=args.extra_features,
    )

    dataset_val = DissolutionDataset(
        data_list_val,
        input_scaler_list,
        preprocess_data_cube_fnc=preprocess_data_cube,
        n_sample_files=VAL_FILES,
        dissolution_steps=dissolution_steps,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        extra_features=args.extra_features,
    )

    delete_arrays([data_list_train, data_list_val])
    
    output_scaler_list = estimate_output_scalers(dataset_train.Y, scaler_type='minmax')
    scale_outputs(dataset_train, output_scaler_list)
    scale_outputs(dataset_val, output_scaler_list)
    
    ckpt_file = args.ckpt_file + '_{epoch:02d}'
    monitored_loss = "Validation Loss" # For all callbacks

    ckpt_params = {
        'monitor': monitored_loss,
        'filename': ckpt_file,
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

    if args.extra_features:
        in_channels = 7
    else:
        in_channels = 4
    
    model_params, model_obj = set_model_params(
        args, in_channels=in_channels, out_channels=len(outputs))

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

    if len(args.model_list) > 0:
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        model_list = [
            model_obj.load_from_checkpoint(f).eval().to(device)
            for f in args.model_list]        

        X_train, Y_train = prepare_multilevel_correction_data(
            dataset_train,
            output_scaler_list,
            model_list,
            device=device
        )

        X_val, Y_val = prepare_multilevel_correction_data(
            dataset_val,
            output_scaler_list,
            model_list,
            device=device
        )

        dataset_train = DissolutionDatasetCorrection(X_train, Y_train)        
        dataset_val = DissolutionDatasetCorrection(X_val, Y_val)        
        delete_arrays([X_train, Y_train, X_val, Y_val])

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    val_loader = DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True)

    model = model_obj(**model_params)
    trainer = L.Trainer(**trainer_params)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]

    parser.add_argument("--model_name", type=str, choices=['convlstm', 'tau', 'ufno'])
    parser.add_argument("--dataset_path", type=str, default="../reactflow/256modelruns")    
    parser.add_argument("--gpu", type=int, default=0, choices=gpu_ids)
    parser.add_argument("--train_files", type=int, default=24)
    parser.add_argument("--val_files", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--outputs", type=str, default='C,eps,Ux,Uy',
                        help="Comma-separated output properties")
    parser.add_argument("--in_steps", type=int, default=5)
    parser.add_argument("--out_steps", type=int, default=5)
    parser.add_argument("--extra_features", action="store_true", default=True)
    parser.add_argument("--ckpt_file", type=str, default='checkpoint')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--model_list", type=str, nargs='+', default=[],
                        help="List of model checkpoints for iterative stacking")
    
    main(parser.parse_args())