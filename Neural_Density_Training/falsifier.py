import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from dreal import *
import matplotlib.patches as patches
from matplotlib import cm

def checkSMTconstraints(vars, a_learnt, divergence_learnt, ball_lb, ball_ub, config, epsilon=0):
    """Check the SMT constraints for the given variables and learnt parameters"""
    ball = Expression(0)
    
    for i in range(len(vars)):
        ball += vars[i]**2
    ball_in_bound = logical_and(ball >= ball_lb, ball <= ball_ub)
    
    a_condition = a_learnt >= 0
    divergence_condition = divergence_learnt > 0
    stability_condition = logical_and(
        logical_imply(ball_in_bound, divergence_condition),
        logical_imply(ball_in_bound, a_condition)
    )
    
    return CheckSatisfiability(logical_not(stability_condition), config)

def AddCounterexamples(x, CE, N, device):
    """Adds counterexamples to the sample set efficiently"""
    nearby = []
    for i in range(CE.size()):
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.random.uniform(lb, ub, N)
        nearby.append(nearby_)
    
    # Vectorized counterexample addition
    new_points = np.column_stack(nearby)
    new_points_tensor = torch.tensor(new_points, dtype=torch.float32, device=device)
    x = torch.cat((x, new_points_tensor), dim=0)
    x = torch.unique(x, dim=0)
    
    return x, new_points_tensor