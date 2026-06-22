import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from dreal import *
import matplotlib.patches as patches
from matplotlib import cm


class SymbolicNN:
    """Convert neural networks to symbolic expressions for dReal"""
    def __init__(self, net):
        self.weights = []
        self.biases = []
        self.extract_params(net)

    def extract_params(self, net):
        for layer in net.net:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.detach().cpu().numpy())
                if layer.bias is not None:
                    self.biases.append(layer.bias.detach().cpu().numpy())

    def forward_symbolic(self, vars):
        z = vars
        for i in range(len(self.weights) - 1):
            z = self._linear_symbolic(z, self.weights[i], self.biases[i])
            z = [tanh(zi) for zi in z]
        z = self._linear_symbolic(z, self.weights[-1], self.biases[-1])
        return z

    def _linear_symbolic(self, x, weight, bias):
        out = []
        for i in range(weight.shape[0]):
            expr = bias[i]
            for j in range(weight.shape[1]):
                expr += weight[i,j] * x[j]
            out.append(expr)
        return out

class activSymbolicNN:
    """Convert neural networks with sigmoid to symbolic expressions"""
    def __init__(self, net):
        self.weights = []
        self.biases = []
        self.extract_params(net)

    def extract_params(self, net):
        for layer in net.net:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.detach().cpu().numpy())
                if layer.bias is not None:
                    self.biases.append(layer.bias.detach().cpu().numpy())

    def forward_symbolic(self, vars):
        z = vars
        for i in range(len(self.weights)):
            z = self._linear_symbolic(z, self.weights[i], self.biases[i])
            z = [(1/(1 + exp(-zi))) for zi in z]
        return z

    def _linear_symbolic(self, x, weight, bias):
        out = []
        for i in range(weight.shape[0]):
            expr = bias[i]
            for j in range(weight.shape[1]):
                expr += weight[i,j] * x[j]
            out.append(expr)
        return out