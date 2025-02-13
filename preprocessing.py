import torch
import numpy as np

from einops import rearrange

MIN_MASK = 1e-12

def box_cox(x, lambda_const=0.01, additive_const=7):
    return additive_const + (x**lambda_const-1)/lambda_const

def inv_box_cox(x, lambda_const=0.01, additive_const=7):
    return ((x-additive_const)*lambda_const+1)**(1/lambda_const)

# dummy function for no scaling
def identity(im):
    return im

def power_n(x, exponent=2):
    return x**exponent

def inv_power_n(x, exponent=2):
    return x**(1/exponent)

def inv_log_transform(im):
    (min, max) = MIN_MASK, 1  # (im[im > 0].min(), im.max())
    return np.exp((im * (np.log(max) - np.log(min))) + np.log(min))

def inv_log_transform_pytorch(im):
    (min, max) = torch.tensor(1e-12), torch.tensor(1)  # (im[im > 0].min(), im.max())
    return torch.exp((im * (torch.log(max) - torch.log(min))) + torch.log(min))

def log_transform(im):
    '''returns log(image) scaled to the interval [0,1]'''
    (min, max) = MIN_MASK, 1  # (im[im > 0].min(), im.max())
    # print(im.min(), im.max())
    return (np.log(np.clip(im, min, max)) - np.log(min)) / (np.log(max) - np.log(min))

def log_transform_pytorch(im):
    '''returns log(image) scaled to the interval [0,1]'''
    (min, max) = MIN_MASK, 1  # (im[im > 0].min(), im.max())
    return (torch.log(torch.clip(im, min, max)) - np.log(min)) / (np.log(max) - np.log(min))


class CustomMinMaxScaler():
    def fit(self, X):
        self.min = X.min()
        self.max = X.max()
        
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X):
        return self.transform(self.fit(X))
    
    def inverse_transform(self, X):
        return (X * (self.max - self.min)) + self.min


class CustomStdScaler():
    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()
        
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        return self.transform(self.fit(X))
    
    def inverse_transform(self, X):
        return (X * self.std) + self.mean


def transform_data(data, scaler_list, axis=-1):
    data = np.moveaxis(data, axis, -1)
    for idx in range(len(scaler_list)):
        data[..., idx] = scaler_list[idx].transform(data[..., idx])
    return np.moveaxis(data, -1, axis)


def inv_transform_data(data, scaler_list, axis=-1):
    data = np.moveaxis(data, axis, -1)
    for idx in range(len(scaler_list)):
        data[..., idx] = scaler_list[idx].inverse_transform(data[..., idx])
    return np.moveaxis(data, -1, axis)


def parse_scaler_type(scaler_type):
    if scaler_type == 'std':
        return CustomStdScaler
    elif scaler_type == 'minmax':
        return CustomMinMaxScaler
    else:
        print("ERROR! Unsupported scaler type:", scaler_type)
        exit(1)


def estimate_input_scalers(data_list, training_steps=101, scaler_type='std'):
    C   = np.stack([data['C'][2:-2, 2:-2, :training_steps] for data in data_list])
    eps = np.stack([data['eps'][2:-2, 2:-2, :training_steps] for data in data_list])
    Ux  = np.stack([data['Ux'][2:-2, 2:-2, :training_steps] for data in data_list])
    Uy  = np.stack([data['Uy'][2:-2, 2:-2, :training_steps] for data in data_list])    
    U = np.sqrt((Ux ** 2) + (Uy ** 2))

    keys = ['C', 'eps', 'Ux', 'Uy', 'U']
    data = [C, eps, Ux, Uy, U]

    scaler_obj = parse_scaler_type(scaler_type)
    scalers_dict = {key: scaler_obj() for key in keys}

    for key, data in zip(scalers_dict.keys(), data):
        scalers_dict[key].fit(data)
    
    return list(scalers_dict.values())


def estimate_output_scalers(data_tensor, axis=-1, scaler_type='minmax'):
    scaler_obj = parse_scaler_type(scaler_type)
    shape = data_tensor.shape
    outputs = shape[axis]
    output_scaler_list = []

    for idx in range(outputs):
        scaler = scaler_obj()
        scaler.fit(np.take(data_tensor, idx, axis=axis))
        output_scaler_list.append(scaler)

    return output_scaler_list


def scale_outputs(dataset, scaler_list, axis=-1):
    dataset.Y = np.moveaxis(dataset.Y, axis, -1)
    for idx in range(len(scaler_list)):
        dataset.Y[..., idx] = scaler_list[idx].transform(dataset.Y[..., idx])
    dataset.Y = np.moveaxis(dataset.Y, -1, axis)


def inv_scale_outputs(dataset, scaler_list, axis=-1):
    dataset.Y = np.moveaxis(dataset.Y, axis, -1)
    for idx in range(len(scaler_list)):
        dataset.Y[..., idx] = scaler_list[idx].inverse_transform(dataset.Y[..., idx])
    dataset.Y = np.moveaxis(dataset.Y, -1, axis)


def preprocess_data_cube_extended(
    data_list, timesteps=100, out_steps=5, outputs=['eps']):
    
    print('Start of preprocess_data_cube')
    X_list = []
    Y_list = []

    for data_dict in data_list:
        C   = data_dict['C'][2:-2, 2:-2, :timesteps]
        eps = data_dict['eps'][2:-2, 2:-2, :timesteps]
        Ux  = data_dict['Ux'][2:-2, 2:-2, :timesteps]
        Uy  = data_dict['Uy'][2:-2, 2:-2, :timesteps]

        Y_out_list = []

        for out in outputs:
            Y_out = data_dict[out][2:-2, 2:-2, out_steps:timesteps].copy()

            if out == 'eps':
                Y_out[Y_out <= MIN_MASK] = 0.0
                
            Y_out_list.append(Y_out)

        X = np.stack([C, eps, Ux, Uy], axis=-1) # (H, W, N, C)
        X = rearrange(X, "H W N C -> N H W C")
        X_list.append(X)
        
        Y = np.stack(Y_out_list, axis=-1) # (H, W, N, C)
        Y = rearrange(Y, "H W N C -> N H W C")
        Y_list.append(Y)

    print('Done preprocess_data_cube')
    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)


