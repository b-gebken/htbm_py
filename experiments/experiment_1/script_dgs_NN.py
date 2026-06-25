# A script for computing the approximated solutions used in the plot scripts via DGS

import numpy as np
import matplotlib.pyplot as plt

from htbm_py.dgs import dgs

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

reg_param = 1*1e-5
N_data = 12
loss_fn_type = 'mse'
problem_data = loss_NN(model,reg_param,N_data,loss_fn_type)

n = problem_data.x0.shape[0]

# np.random.seed(0)
# problem_data.x0 = 10*np.random.rand(n) - 5
problem_data.x0 = np.array([ 1.8707043710581506 ,  0.9797012580637289 , -1.149984375818818  ,
       -0.9529418297280058 ,  0.13379081598724818,  0.3356932320611862 ,
       -1.6547984544866747 ,  0.33263217462235534, -0.6990767522803849 ,
       -0.10888924850817888,  0.6618606222814275 , -0.5939706839051107 ,
        0.6207110700105015 ,  0.8335780531871593 , -2.004238301055387  ,
       -1.4558886451760369 ,  1.9150017131588786 ]) + \
        0*1e-1 * (2*np.random.rand(n)-1)

## Set the parameters for the method #####

j_max = 7
kappa_eps = 0.1
eps0 = 1

algo_options = {
    'eps_arr': eps0*kappa_eps**(np.arange(j_max)), # different to Matlab because index offset
    'delta_arr': 1e-5 * np.ones(j_max),
    'rand_sample_N': 1,
    'memory_max_size': 0,
    'c': 0.5,
    'ls_flag': 'armijo',
    'max_iter': 10000,
    'disp_flag': 3
}

## Run DGS ###############################

x_opt, f_opt, x_arr_list, fx_arr_list, eval_counter, num_sample_arr_list = dgs(problem_data,algo_options)

## Plots #################################

np.set_printoptions(precision=50)
print('loss unreg: ',loss_unreg(x_opt,model,N_data,loss_fn_type))

numiter_arr = [len(fx_arr) for fx_arr in fx_arr_list]
num_total_iter = sum(numiter_arr)

fx_all = np.zeros(num_total_iter)
num_sample_all = np.zeros(num_total_iter)
counter = 0
for j in range(j_max):
    fx_all[counter:counter+len(x_arr_list[j])] = np.asarray(fx_arr_list[j])
    num_sample_all[counter:counter+len(x_arr_list[j])] = np.asarray(num_sample_arr_list[j])
    counter = counter + len(x_arr_list[j])

    # (1,1) ##############################

ax1 = plt.subplot(2,3,1)
ax1.plot(np.log10(fx_all[:-1] - np.min(fx_all)),'k.-')
ax1.grid()

    # (1,2) ##############################

ax2 = plt.subplot(2,3,2)
ax2.plot(np.log10(fx_all),'k.-')
ax2.grid()

    # (1,3) ##############################

ax3 = plt.subplot(2,3,3)
ax3.plot(numiter_arr,'k.-')
ax3.grid()

    # (2,1) ##############################

ax4 = plt.subplot(2,3,4)
ax4.plot(num_sample_all,'k.-')
ax4.grid()

plt.show()

print('x_opt = ',repr(x_opt))

visualize(x_opt,model,N_data)