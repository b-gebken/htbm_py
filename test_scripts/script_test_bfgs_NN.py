# A script for training a neural network via BFGS

import numpy as np

from htbm_py.bfgs import bfgs

## Define problem ########################

from torch import nn
from htbm_py.test_functions.loss_NN import loss_NN
from htbm_py.test_functions.loss_NN import loss_unreg
from htbm_py.test_functions.loss_NN import visualize
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to("cpu")

reg_param = 0.0001
N_data = 20
loss_fn_type = 'mse'
problem_data = loss_NN(model,reg_param,N_data,loss_fn_type)

n = problem_data.x0.shape[0]

# np.random.seed(0)
# problem_data.x0 = 10*np.random.rand(n) - 5
problem_data.x0 = np.array([-1.2737644744674492e+00,  1.1316896534481780e+00,
       -6.0802500315902586e-01,  1.1045578177326596e+00,
        2.3678135927411392e-01, -3.2709127111716396e-01,
       -1.6562065817714948e+00, -6.9833969169552712e-01,
       0,  7.2211570940229886e-01,
       -6.7013132103765727e-01, -9.3511627427513966e-01,
        1.2770464483111723e+00, -6.8141573028809174e-02,
        3.6315927217865878e+00,  2.3955048787183748e+00,
       -1.3858824895083590e+00]) + 1e-0 * np.random.rand(n)

## Set the parameters for the method #####

algo_options = {
    'N_iter': 10000,
    'H0': np.eye(n),
    'c1': 0.0001,
    'c2': 0.5,
    'reset_period': np.inf,
    'step_threshold': 1e-10,
    'descent_threshold': 1e-10,
    'disp_flag': 2
}

## Run BFGS ##############################

result_bfgs = bfgs(problem_data,algo_options)

## Plots #################################

x_arr = result_bfgs['x_arr']
print('x_final = ',repr(x_arr[-1]))

print('Unregularized loss: ',loss_unreg(x_arr[-1],model,N_data,loss_fn_type))

# eigvals, eigvecs = np.linalg.eig(result_bfgs['H_arr'][-1])
# print('Sorted Eigenvalues: ',np.sort(eigvals))

visualize(x_arr[-1],model,N_data)