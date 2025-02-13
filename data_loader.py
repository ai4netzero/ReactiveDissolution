import numpy as np
import h5py
import torch
import os

from einops import rearrange
from torch.utils.data import Dataset
from preprocessing import *


def load_data(file_name):
    print(f"Loading the file: {file_name}")
    data_dict = {}

    with h5py.File(file_name, "r") as file_handle:
        # List all groups
        print(f'Keys: {file_handle.keys()}')
        keys = list(file_handle.keys())
        for key in keys:
            try:
                data_dict[key] = np.array(file_handle[key])
            except MemoryError:
                print(f"MemoryError: Could not load the variable {key} due to insufficient memory.")
            
            print(f'Done loading the variable {key} of shape: {data_dict[key].shape}')

        print(f'Done with {file_name} == closing file now')

    return data_dict


def read_full_dataset(dataset_files, dataset_path="../reactflow/256modelruns"):
    data_list = []
    
    for hdf in dataset_files:
        filename_hdf = os.path.join(dataset_path, hdf)
        print(filename_hdf)
        
        data_dict = load_data(filename_hdf)
        data_list.append(data_dict)

    return data_list


class DissolutionDataset(Dataset):
    def __init__(
        self,
        data_list,
        input_scaler_list,
        preprocess_data_cube_fnc,
        n_sample_files,
        dissolution_steps=100,
        in_steps=5,
        out_steps=5,
        extra_features=True
    ):
        self.input_scaler_list = input_scaler_list
        self.n_sample_files = n_sample_files
        self.dissolution_steps = dissolution_steps
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.extra_features = extra_features

        seq_steps = self.in_steps + self.out_steps
        self.steps_per_sample = self.dissolution_steps - seq_steps + 1

        self.X, self.Y = preprocess_data_cube_fnc(data_list)

    def __getitem__(self, idx):
        file_idx = idx // self.steps_per_sample
        timestep = idx % self.steps_per_sample

        X_start = self.dissolution_steps * file_idx + timestep
        Y_start = (self.dissolution_steps - self.out_steps) * file_idx + timestep

        X_end = X_start + self.in_steps
        Y_end = Y_start + self.out_steps

        X = self.X[X_start : X_end]
        Y = self.Y[Y_start : Y_end]

        X = torch.FloatTensor(rearrange(X, "T H W C -> T C H W"))
        Y = torch.FloatTensor(rearrange(Y, "T H W C -> T C H W"))

        X = transform_inputs(X, self.input_scaler_list, self.extra_features)
        return X, Y

    def __len__(self):
        return self.n_sample_files * self.steps_per_sample


# Ensure that the inputs are in their ORIGINAL SCALES
def transform_inputs(inputs, input_scaler_list, extra_features=False):
    if extra_features:
        C   = inputs[:, 0]
        eps = inputs[:, 1]
        Ux  = inputs[:, 2]
        Uy  = inputs[:, 3]
    
        # Engineered features
        eps_filter = (eps > 0.01) * (eps < 0.99)
        c_filter = C > 0.01/100
    
        combined_filter = (eps_filter * c_filter)        
        C_scaled = 2 * log_transform(C) - 0.5
        U = (np.sqrt((Ux ** 2) + (Uy ** 2)))
    
        stack = torch.stack([U, C_scaled, combined_filter], dim=1)
        new_inputs = torch.cat([inputs, stack], dim=1)
        N = 5
        
    else:
        new_inputs = inputs.clone()
        N = 4
        
    for k, scaler in enumerate(input_scaler_list[:N]):
        new_inputs[:, k] = scaler.transform(new_inputs[:, k])

    return new_inputs
    
# "preds" must be SCALED
def get_next_inputs(preds, input_scaler_list, output_scaler_list, extra_features):
    preds_inv = torch.empty_like(preds)

    for k, scaler in enumerate(output_scaler_list):
        preds_inv[:, k] = scaler.inverse_transform(preds[:, k])
    
    return transform_inputs(preds_inv, input_scaler_list, extra_features)


def prepare_multilevel_correction_data(
    base_dataset : DissolutionDataset,
    output_scaler_list,
    model_list=[],
    device="cpu"):

    print('Start of prepare_multilevel_correction_data')
    X_list = []
    Y_list = []

    input_scaler_list = base_dataset.input_scaler_list

    for idx in range(len(base_dataset)): 
        X, Y = base_dataset[idx]

        for model in model_list:
            approx = model(
                torch.FloatTensor(X)
                .to(device)
                .unsqueeze(0)).squeeze().detach().cpu()
            
            X = get_next_inputs(
                approx,
                input_scaler_list,
                output_scaler_list,
                base_dataset.extra_features)
    
        X_list.append(X)
        Y_list.append(Y)

    print('Done prepare_multilevel_correction_data')
    return torch.stack(X_list), torch.stack(Y_list)


class DissolutionDatasetCorrection(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __getitem__(self, idx):
        X, Y = self.X[idx].clone(), self.Y[idx].clone()
        return X, Y        

    def __len__(self):
        return self.X.shape[0]
