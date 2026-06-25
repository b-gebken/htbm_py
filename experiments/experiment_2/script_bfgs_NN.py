# A script for computing the approximated solutions used in the plot scripts via BFGS

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

reg_param = 0*1e-5
N_data = 20
loss_fn_type = 'mse'
problem_data = loss_NN(model,reg_param,N_data,loss_fn_type)

n = problem_data.x0.shape[0]

# np.random.seed(1)
# problem_data.x0 = 10*np.random.rand(n) - 5
problem_data.x0 = np.array([ 1.702609416959709  ,  1.061145217498979  , -0.7227144799889186 ,
       -0.8870194166117329 ,  0.14079184940494102,  0.2232186951682984 ,
       -2.157980629689821  ,  0.7242306201951149 , -0.9590041800524562 ,
        1.2791650531062513 , -0.493628101625141  , -0.8116430457736651 ,
        0.5075063561607176 ,  0.7248626079837421 , -2.6889531716526505 ,
       -1.8225216625095735 ,  1.9150866605111143 ]) + \
        1e-1 * (2*np.random.rand(n)-1)

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
np.set_printoptions(precision=20)
print('x_final = ',repr(x_arr[-1]))

print('Unregularized loss: ',loss_unreg(x_arr[-1],model,N_data,loss_fn_type))

# eigvals, eigvecs = np.linalg.eig(result_bfgs['H_arr'][-1])
# print('Sorted Eigenvalues: ',np.sort(eigvals))

visualize(x_arr[-1],model,N_data)