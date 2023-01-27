from typing import Union


import torch
from torch import nn
import numpy as np

activ = Union[str, nn.Module]

str_to_activtion = {
    "relu" , nn.ReLU(),
    "tanh" , nn.Tanh(),
    "sigmoid" , nn.Sigmoid(),
    "selu" , nn.SELU(),
    "softplus", nn.Softplus(),
    "identity", nn.Identity(),
    "leaky_relu", nn.LeakyReLU()
}

device = None

def build_nn(
        input_size: int,
        output_size: int,
        num_layers: int,
        size: int,
        activation: activ = "tanh",
        output_activation: activ = "identity"
):
    
    if isinstance(activation, str):
        activation = str_to_activtion[activation]
    if isinstance(output_activation, str):
        output_activation = str_to_activtion[activation]

    layers = []

    in_size = input_size

    for _ in range(num_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size

    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    fcn  = nn.Sequential(*layers)
    fcn.to(device)

    return fcn


def gpu_init(use_gpu = True, gpu_id = 0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda: " + str(gpu_id))
        print("GPU available! Using GPU id {}".format(gpu_id))

    else:
        device = torch.device("cpu")
        print("GPU not found! Using CPU instead")

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)              


def from_numpy(input : Union[np.ndarray, dict], **kwargs):

    if isinstance(input, dict):
        return {k: from_numpy(v) for k,v in input.items()}
    
    else:
        input = torch.from_numpy(input, **kwargs)
        if input.dtype == torch.float64:
            input = input.float()

        return input.to(device)
    

def to_numpy(tensor: Union[torch.tensor, dict], **kwargs):

    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k,v in tensor.items()}
    
    else:
        return tensor.to("cpu").detach().numpy()
    
    


