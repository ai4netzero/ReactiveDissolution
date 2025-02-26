# Machine Learning for Reactive Dissolution in Porous Media

## Introduction

This repository contains a PyTorch (Lightning) implementation of three different deep learning algorithms to forecast the future states of reactive dissolution in porous media. It includes the following algorithms (with links to their respective original repositories):

- [Convolutional Long Short-Term Memory (ConvLSTM)](https://github.com/ndrplz/ConvLSTM_pytorch)
- [U-Shaped Fourier Neural Operator (U-FNO)](https://github.com/gegewen/ufno)
- [Temporal Attention Unit (TAU)](https://github.com/chengtan9907/OpenSTL)

## Main Requirements

- PyTorch-Lightning
- OpenSTL (for TAU)
- NumPy
- Pandas
- Argparse
- Scikit-Learn
- Jupyter

Please refer to [requirements.txt](requirements.txt) for more details.

## Overview

Given the numbers of input time steps $T_{in}$ and output time steps $T_{out}$, this code trains a deep learning model which receives a sequence of dissolution states of shape $(T_{in}, C_{in}, H_{in}, W_{in})$ and outputs a sequence of forecasted states of shape $(T_{out}, C_{out}, H_{out}, W_{out})$. In essence, the properties comprised by the input features $C_{in}$ and the output features $C_{out}$ are defined as follows:

- $\mathbf{C}$: the concentration of the acidic solution used in the dissolution;
- $\mathbf{eps}$: the volume fraction of the pore space occupied by pore in each voxel;
- $\mathbf{U_x}$: the magnitude and direction of flow of the acidic solution in the horizontal axis of a 2-D Cartesian plane;
- $\mathbf{U_y}$: same as $\mathbf{U_x}$, but for the vertical axis.

Aside from those, there are three (optional) engineered features to be included as input features:

- **Magnitude of Velocity**: $\mathbf{U} = \sqrt{\mathbf{U_x}^2 + \mathbf{U_y}^2}$;
- **C Scaled**: a log-transformation on the $\mathbf{C}$ feature;
- **Combined Filter**: a binary mask based on $\mathbf{C}$ and $\mathbf{eps}$ constraints which shows the portions of the grains in a porous matrix that are being dissolved at a particular time step $t$.

Originally, our dataset considers input state maps $(H_{in}, W_{in}) = (256, 256)$ and output state maps $(H_{out}, W_{out}) = (256, 256)$, but the code is flexible for training at different input / output shapes, as well as performing inference at higher resolutions than those used in the training.

## Training

The general settings to train a deep learning model can be found at the main function in the [train.py](train.py) script. The specific parameters for each model can be configured in the [config.py](config.py) script. If one wants to implement a new model (_Hint_: use the Base_Model class from [models/base_model.py](models/base_model.py)), they should also include an option to that model in the "--model_name" option, as well as the constructor parameters in [config.py](config.py).

Example command for training a model (Level 0 Prediction) with the full dataset:

```
python train.py --model_name tau --dataset_path <path/to/your/dataset> --ckpt_prefix tau_level_0 > train_level_0.log
```

All checkpoints created by Lightning are stored in tb_logs/<model_name> folder. Then, to train the next levels of the iterative stacking, each level must be trained separately, and the models that have been trained before a level must be passed as arguments in the command line in the exact order.

Example for training a Level 1 TAU model:

```
python train.py --model_name tau --dataset_path <path/to/your/dataset> --ckpt_prefix tau_level_1 --model_list <path/to/level/0> > train_level_1.log
```

Same rationale for Level 2:

```
python train.py --model_name tau --dataset_path <path/to/your/dataset> --ckpt_prefix tau_level_2 --model_list <path/to/level/0> <path/to/level/1> > train_level_2.log
```

> [!IMPORTANT]
> Make sure that the general settings and the model parameters are exactly the same for all levels.

## Evaluation & Visualization

TBD

## Acknowledgements

This work is funded by the Engineering and Physical Sciences Research Council's ECO-AI Project grant (reference number EP/Y006143/1), with additional financial support from the PETRONAS Centre of Excellence in Subsurface Engineering and Energy Transition (PACESET).

## Citation

TBD
