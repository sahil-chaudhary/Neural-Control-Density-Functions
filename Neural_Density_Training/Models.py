# This file takes care of the model definitions required for the training of the neural density controller.import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from dreal import *
import matplotlib.patches as patches
from matplotlib import cm


class Polynomial_net(nn.Module):
    def __init__(self, layers, input_dim, hidden_dim, output_dim):
        """Initializes the polynomial neural network. The args means:
        - layers: number of layers in the network
        - input_dim: dimension of the input
        - hidden_dim: dimension of the hidden layers
        - output_dim: dimension of the output
        """
        super(Polynomial_net, self).__init__()
        
        # if the number of layers is 2, then it means we will have linear, tanh, linear layers. If the number of layers is 3, then it means we will have linear, tanh, linear, tanh, linear layers and so on.

        assert layers >= 2, "The number of layers must be at least 2"
        layer_list = []

        # First Layer
        layer_list.append(nn.Linear(input_dim, hidden_dim))

        # Intermediate Hidden Layers : hidden to hidden
        for _ in range(layers-2):
            layer_list.append(nn.Tanh())
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))

        # Output Layer
        layer_list.append(nn.Tanh())
        layer_list.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layer_list)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the weights of the network using Xavier initialization."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Defines the forward pass of the network."""
        return self.net(x)
    
class Polynomial_net_bias_zero(nn.Module):
    def __init__(self, layers, input_dim, hidden_dim, output_dim):
        super(Polynomial_net_bias_zero, self).__init__()

        assert layers >= 2, "The number of layers must be at least 2"
        layer_list = []

        # First Layer
        layer_list.append(nn.Linear(input_dim, hidden_dim))

        # Intermediate Hidden Layers : hidden to hidden
        for _ in range(layers-2):
            layer_list.append(nn.Tanh())
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))

        # Output Layer
        layer_list.append(nn.Tanh())
        layer_list.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layer_list)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the weights of the network using Xavier initialization and sets biases to zero."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                m.bias.requires_grad = False

    def forward(self, x):
        """Defines the forward pass of the network."""
        return self.net(x)
    
class Numerator_polynomial_net(nn.Module):
    def __init__(self, layers, input_dim, hidden_dim, output_dim):
        super(Numerator_polynomial_net, self).__init__()

        assert layers >= 2, "The number of layers must be at least 2"
        layer_list = []

        # First Layer
        layer_list.append(nn.Linear(input_dim, hidden_dim))

        # Intermediate Hidden Layers : hidden to hidden
        for _ in range(layers-2):
            layer_list.append(nn.Sigmoid())
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))

        # Output Layer
        layer_list.append(nn.Sigmoid())
        layer_list.append(nn.Linear(hidden_dim, output_dim))
        layer_list.append(nn.Sigmoid())

        self.net = nn.Sequential(*layer_list)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the weights of the network using Xavier initialization."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            
    def forward(self, x):
        """Defines the forward pass of the network."""
        return self.net(x)
    
class GammaFunctionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GammaFunctionNet, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        nn.init.constant_(self.linear_layer.weight, 0.5)
        nn.init.constant_(self.linear_layer.bias, (1e-20))


    def forward(self, x):
        if x.dim() == 1:
            return abs(self.linear_layer.bias)
        return abs(self.linear_layer.bias.expand(x.shape[0], -1))

