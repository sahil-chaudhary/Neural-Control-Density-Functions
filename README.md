# Neural Control Density Function
The repo includes experiments related to Neural Control Density Function. We consider rantzer's examples for certifying the stability which are demonstrated in:
1. example4.ipynb: The notebook includes the functions to train a density function as well as lyapunov function. The weights are stored in Model\_paths and has been used in cells to plot the phase portrait.  
2. example5.ipynb: The notebook includes the functions to train a density function as well as lyapunov function. The weights are stored in Model\_paths and has been used in cells to plot the phase portrait.

For the problem of control Synthesis, we consider examples which includes:
1. prajna1.ipynb
2. prajna2.ipynb
3. prajna3.ipynb
4. Van der Pol Oscillator.ipynb
To demonstrate the effectiveness of our parameterization, we train a vanilla density-controller pair for 1st example in the paper. By vanilla, we mean the parameterization only considers $\rho=a(x)/b(x)$ and $\psi(x)=c(x)/b(x)$. The notebook corresponding to this is included in:
1. prajna_1_without_parameterization.ipynb

For the problem of mixing/blending, we consider the examples of inverted pendulum and path following. The notebooks corresponding to that are:
1. optimized_inverted_pendulum.ipynb
2. path_following.ipynb

To demonstrate the safe-stable control synthesis, we utilize a variant of duffing oscillator [just to ensure the point of stability is at origin] and reconsider prajna1 example with an unsafe set described in paper. The notebooks corresponding to safety are:
1. duffing_oscillator_safety.ipynb
2. prajna_safety.ipynb

To demonstrate the instability analysis, we also consider a system with an unstable equilibiria:
1. unstable_equil.ipynb
