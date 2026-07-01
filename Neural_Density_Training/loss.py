import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from dreal import *
import matplotlib.patches as patches
from matplotlib import cm

# This file contains the loss functions used for training the neural network. We are going to take:
# 1. x: collection of the state of the system
# 2. takes the dynamics of the system as f_net, g_net, div_f_net, div_g_net; these are functions 
# 3. network a_net, b_net, c_net
class stability_loss(nn.Module):
    def __init__(self,
                 f,
                 g,
                 div_f,
                 div_g):
        super().__init__()
        self.f = f
        self.g = g
        self.div_f = div_f
        self.div_g = div_g

    def forward(self, x, a_net, b_net, c_net,gamma_net):
        divergence = self._control_density_risk(x, a_net, b_net, c_net)
        gamma_vals = gamma_net(x).squeeze()
        loss = F.relu(gamma_vals-divergence ).mean()
        a_loss = F.relu(-a_net(x)).mean()   
        gamma_loss = F.relu(-gamma_vals).mean()

        return loss + a_loss + gamma_loss
    
    # helpers

    def _gradient(self, func: callable, x: torch.Tensor):
        """Computes the gradient of the function fun with respect to x."""        
        x = x.clone().detach().requires_grad_(True)
        output = func(x).sum()
        return torch.autograd.grad(output, x, create_graph=True)[0]
    
    def _gradient_norm_sq(self, func: callable, x: torch.Tensor):
        """Computes the gradient of the squared norm of function fun with respect to x."""
        x = x.clone().detach().requires_grad_(True)
        output = func(x)
        norm_sq = (output ** 2).sum(dim=1).sum()
        return torch.autograd.grad(norm_sq, x, create_graph=True)[0]

    def _control_density_risk(self, x, a_net, b_net, c_net):
        """Computes the control density risk for the given state x and networks a_net, b_net, c_net."""

        # --- values --------------------------------------------------
        a_vals = a_net(x).squeeze()          # [B]
        c_vals = c_net(x).squeeze()          # [B]
        f_vals = self.f(x)                   # [B, 2]
        g_vals = self.g(x)                   # [B, 2]

        # --- divergences ---------------------------------------------
        div_f_vals = self.div_f(x)           # [B]
        div_g_vals = self.div_g(x)           # [B]

        # --- gradients -----------------------------------------------
        grad_a = self._gradient(a_net, x)           # [B, 2]
        grad_c = self._gradient(c_net, x)           # [B, 2]
        grad_b = self._gradient_norm_sq(b_net, x)   # [B, 2]

        # --- term1: divergence terms ---------------------------------
        term1 = (
            a_vals * div_f_vals
            + div_g_vals * c_vals
            + (grad_a * f_vals).sum(dim=1)
            + (grad_c * g_vals).sum(dim=1)
        )

        # --- term2: inner product terms ------------------------------
        scaled_x = 2 * x + grad_b                              # [B, 2]
        combined = (
            f_vals * a_vals.unsqueeze(1)
            + g_vals * c_vals.unsqueeze(1)
        )                                                       # [B, 2]
        term2 = (scaled_x * combined).sum(dim=1)

        return term1 - term2
    

class invariance_loss(nn.Module):
    def __init__(self,
                 f,
                 g,
                 div_f,
                 div_g):
        super().__init__()
        self.f = f
        self.g = g
        self.div_f = div_f
        self.div_g = div_g

    def forward(self, x, a_net, b_net, c_net,tau_net):    
        divergence_term = self.div_dynamics(x, a_net, b_net, c_net, tau_net)
        loss = F.relu(-divergence_term).mean()

        return loss
    def _gradient(self, func: callable, x: torch.Tensor):
        """Computes the gradient of the function fun with respect to x."""        
        x = x.clone().detach().requires_grad_(True)
        output = func(x).sum()
        return torch.autograd.grad(output, x, create_graph=True)[0]
    
    def _gradient_norm_sq(self, func: callable, x: torch.Tensor):
        """Computes the gradient of the squared norm of function fun with respect to x."""
        x = x.clone().detach().requires_grad_(True)
        output = func(x)
        norm_sq = (output ** 2).sum(dim=1).sum()
        return torch.autograd.grad(norm_sq, x, create_graph=True)[0]    
    def div_dynamics(self, x, a_net, b_net, c_net,tau_net):

        # --- values------------------------------------------------
        a_vals = a_net(x).squeeze()          # [B]
        c_vals = c_net(x).squeeze()          # [B]
        f_vals = self.f(x)                   # [B, 2]
        g_vals = self.g(x)                   # [B, 2]

        # --- divergences -------------------------------------------
        div_f_vals = self.div_f(x)           # [B]
        div_g_vals = self.div_g(x)           # [B]

        # --- gradients ---------------------------------------------
        grad_a = self._gradient(a_net, x)           # [B, 2]
        grad_c = self._gradient(c_net, x)           # [B, 2]
        grad_b = self._gradient_norm_sq(b_net, x)
        
        # \grad u = (a \grad c - c \grad a) / a^2
        numerator = a_vals.unsqueeze(1) * grad_c - c_vals.unsqueeze(1) * grad_a
        denominator = a_vals.unsqueeze(1) ** 2 + 1e-8
        grad_u = numerator / denominator

        # \div(f + gu) = div(f) + div(g)u + g \cdot \grad u
        div_f_plus_gu = div_f_vals + div_g_vals * (c_vals / a_vals) + (g_vals * grad_u).sum(dim=1)

        # exponent is \exp(||x||^2 + ||b||^2)
        exponent_term = torch.exp(torch.sum(x**2, dim=1) + torch.sum(b_net(x)**2, dim=1))
        tau_vals = tau_net(x).squeeze()
        multiplier = a_vals - tau_vals * exponent_term

        # term 1: (a - tau * exp) * div(f + gu)
        term1 = multiplier * div_f_plus_gu

        # term 2: (f+gu)^T (\grad a - a(2x+ \grad ||b||^2))
        f_plus_gu = f_vals + g_vals * (c_vals / a_vals).unsqueeze(1)
        scaled_x = 2 * x + grad_b
        grad_a_scaled = grad_a - a_vals.unsqueeze(1) * scaled_x
        term2 = (f_plus_gu * grad_a_scaled).sum(dim=1)

        return term1 + term2


class safety_loss(nn.Module):
    def __init__(self,
                 f,
                 g):
        super().__init__()
        self.f = f
        self.g = g
        pass
