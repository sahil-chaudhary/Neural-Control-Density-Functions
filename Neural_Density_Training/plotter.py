import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from dreal import *
import matplotlib.patches as patches
from matplotlib import cm
import torch
from matplotlib import _cm
from matplotlib.patches import Circle

def plot_contour_density(f,g,a,b,c,x_lim,y_lim,density_threshold,title,x_axis_title,y_axis_title,device,plot_flow=True):
    """Plots the contour of the density function defined by a, b, c,f,g"""
    x1 = np.linspace(x_lim[0], x_lim[1], 200)
    x2 = np.linspace(y_lim[0], y_lim[1], 200)
    X,Y = np.meshgrid(x1,x2)

    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]),dtype=torch.float32).to(device)

    with torch.no_grad():
        rho =  a(grid_points) / (
            torch.exp(torch.sum(grid_points**2, dim=1, keepdim=True)
                     + torch.sum((b(grid_points))**2, dim=1, keepdim=True)))
        rho_max = rho.max().item()
        rho = rho / rho_max  
        
    rho = rho.squeeze().cpu().numpy().reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    filled_contours = ax.contourf(X,Y,rho,alpha=0.3,cmap=cm.coolwarm)
    cbar = plt.colorbar(filled_contours,ax=ax)
    cbar.set_label('Density Value', fontsize = 16)

    # Draw the RoA
    ax.contour(X,Y,rho-density_threshold, levels=0, colors='black', linewidths=2)

    if plot_flow:
        x1_flow = np.linspace(x_lim[0], x_lim[1], 20)
        x2_flow = np.linspace(y_lim[0], y_lim[1], 20)
        X_flow, Y_flow = np.meshgrid(x1_flow, x2_flow)
        flow_points = torch.tensor(np.column_stack([X_flow.ravel(), Y_flow.ravel()]), dtype=torch.float32).to(device)
        with torch.no_grad():
            rho_flow = a(flow_points) / (
                torch.exp(torch.sum(flow_points**2, dim=1, keepdim=True)
                         + torch.sum((b(flow_points))**2, dim=1, keepdim=True)))
            u_values = c(flow_points)/a(flow_points)
            
        DX, DY = np.zeros_like(X_flow), np.zeros_like(Y_flow)
        for i in range(len(x1_flow)):
            for j in range(len(x2_flow)):
                idx = i * len(x1_flow) + j
                x = torch.tensor([X_flow[i, j], Y_flow[i, j]], device=device)
                f_val = f(x)
                g_val = g(x)
                u = u_values[idx]
                flow = f_val + g_val * u
                DX[i, j], DY[i, j] = flow[0].cpu().numpy(), flow[1].cpu().numpy()        

        magnitude = np.sqrt(DX**2 + DY**2)
        DX, DY = DX / magnitude, DY / magnitude

        ax.streamplot(X_flow, Y_flow, DX, DY, color='gray',
                     linewidth=0.5, density=1.0,
                     arrowstyle='-|>', arrowsize=1.5)


    circle = Circle((0 , 0), abs(x_lim[0]),  color='red', fill=False, 
                   linewidth=2, linestyle='--', label='Valid Region')     
    ax.add_patch(circle)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # 2. Ensure a square aspect ratio so the circle isn't an oval
    ax.set_aspect('equal')
    ax.set_xlabel(x_axis_title, fontsize=16)
    ax.set_ylabel(y_axis_title, fontsize=16)
    ax.set_title(title, fontsize=18)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='NCDF'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Valid Region')   
    ]
    ax.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),   # x=centre, y=below the axes
    ncol=2,                         # both entries side by side
    fontsize=13,
    frameon=True,
    )
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_contour_mixed_density(f,g,a,b,c,a2,b2,c2,x_lim,y_lim,density_threshold1,density_threshold2,density_threshold, title,x_axis_title,y_axis_title, device, plot_flow=True):
    """ Plots the mixed contour of the density function defined by a, b, c,f,g and a2,b2,c2"""
    x1 = np.linspace(x_lim[0], x_lim[1], 200)
    x2 = np.linspace(y_lim[0], y_lim[1], 200)
    X,Y = np.meshgrid(x1,x2)

    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]),dtype=torch.float32).to(device)

    with torch.no_grad():
        rho1 = a(grid_points) / (
            torch.exp(torch.sum(grid_points**2, dim=1, keepdim=True)
                     + torch.sum((b(grid_points))**2, dim=1, keepdim=True)))
        rho1_max = rho1.max().item()
        rho1 = rho1 / rho1_max
        rho2 = a2(grid_points) / (
            torch.exp(torch.sum(grid_points**2, dim=1, keepdim=True)
                     + torch.sum((b2(grid_points))**2, dim=1, keepdim=True)))
        rho2_max = rho2.max().item()
        rho2 = rho2 / rho2_max

        rho = rho1 + rho2
        rho1 = rho1.squeeze().cpu().numpy().reshape(X.shape)
        rho2 = rho2.squeeze().cpu().numpy().reshape(X.shape)
        rho = rho.squeeze().cpu().numpy().reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    filled_contours = ax.contourf(X,Y,rho,alpha=0.3,cmap=cm.coolwarm)
    cbar = plt.colorbar(filled_contours,ax=ax)
    cbar.set_label('Mixed Density Value', fontsize = 16)

    # Draw the RoA
    ax.contour(X,Y,rho-density_threshold, levels=0, colors='black', linewidths=2,linestyles='-')
    ax.contour(X,Y,rho1-density_threshold1, levels=0, colors='blue', linewidths=2, linestyles='--')
    ax.contour(X,Y,rho2-density_threshold2, levels=0, colors='green', linewidths=2, linestyles='--')

    if plot_flow:
        x1_flow = np.linspace(x_lim[0], x_lim[1], 20)
        x2_flow = np.linspace(y_lim[0], y_lim[1], 20)
        X_flow, Y_flow = np.meshgrid(x1_flow, x2_flow)
        flow_points = torch.tensor(np.column_stack([X_flow.ravel(), Y_flow.ravel()]), dtype=torch.float32).to(device)
        with torch.no_grad():
            rho_flow1 = a(flow_points) / (
                torch.exp(torch.sum(flow_points**2, dim=1, keepdim=True)
                         + torch.sum((b(flow_points))**2, dim=1, keepdim=True)))
            rho_flow1_max = rho_flow1.max().item()
            rho_flow1 = rho_flow1 / rho_flow1_max
            u_values1 = c(flow_points)/a(flow_points)
            rho2_flow = a2(flow_points) / (
                torch.exp(torch.sum(flow_points**2, dim=1, keepdim=True)
                         + torch.sum((b2(flow_points))**2, dim=1, keepdim=True)))
            rho2_flow_max = rho2_flow.max().item()
            rho2_flow = rho2_flow / rho2_flow_max
            u_values2 = c2(flow_points)/a2(flow_points)

            u_values = (rho_flow1 * u_values1 + rho2_flow * u_values2) / (rho_flow1 + rho2_flow)
        
        DX, DY = np.zeros_like(X_flow), np.zeros_like(Y_flow)
        for i in range(len(x1_flow)):
            for j in range(len(x2_flow)):
                idx = i * len(x1_flow) + j
                x = torch.tensor([X_flow[i, j], Y_flow[i, j]], device=device)
                f_val = f(x)
                g_val = g(x)
                u = u_values[idx]
                flow = f_val + g_val * u
                DX[i, j], DY[i, j] = flow[0].cpu().numpy(), flow[1].cpu().numpy()        

        magnitude = np.sqrt(DX**2 + DY**2)
        DX, DY = DX / magnitude, DY / magnitude

        ax.streamplot(X_flow, Y_flow, DX, DY, color='gray',
                     linewidth=0.5, density=1.0,
                     arrowstyle='-|>', arrowsize=1.5)

    circle = Circle((0 , 0), abs(x_lim[0]),  color='red', fill=False,
                     linewidth=2, linestyle='--', label='Valid Region')
    ax.add_patch(circle)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # 2. Ensure a square aspect ratio so the circle isn't an oval
    ax.set_aspect('equal')
    ax.set_xlabel(x_axis_title, fontsize=16)
    ax.set_ylabel(y_axis_title, fontsize=16)
    ax.set_title(title, fontsize=18)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Mixed NCDF'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='NCDF 1'),
        Line2D([0], [0], color='green', lw=2, linestyle='--', label='NCDF 2'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Valid Region')   
    ]
    ax.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),   # x=centre, y=below the axes
    ncol=2,                         # both entries side by side
    fontsize=13,
    frameon=True,
    )
    plt.tight_layout()
    plt.show()
    return fig, ax
