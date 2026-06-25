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

reg_param = 0*1e-4
N_data = 100
loss_fn_type = 'mae'
problem_data = loss_NN(model,reg_param,N_data,loss_fn_type)

n = problem_data.x0.shape[0]

# np.random.seed(0)
# problem_data.x0 = 10*np.random.rand(n) - 5
problem_data.x0 = np.array([ 2.299430584734672  ,  1.2166667699741789 , -1.117349383845249  ,
       -1.167651988153054  ,  0.19315048926173387,  0.3348071765023482 ,
       -2.1499285783524766 ,  0.9216049699702981 , -0.7644850680725641 ,
        1.1414771750236832 , -0.44625415956890796, -0.7879940806938198 ,
        0.6503472015193431 ,  1.0429861152832838 , -1.858151417714012  ,
       -1.197417071571756  ,  1.8826460041890773 ])

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