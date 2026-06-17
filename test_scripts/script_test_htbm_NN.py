import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from test_functions.simple_NN import simple_NN
from test_functions.simple_NN import visualize
from test_functions.simple_NN import loss_unreg

from htbm_py.htbm import local_method

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
problem_data = simple_NN(model,0.0001)

n = problem_data.x0.shape[0]

# np.random.seed(0)
# x = np.random.rand(17)
# print(problem_data.oracle[0](x))
# print(problem_data.oracle[1](x))
# print(problem_data.oracle[2](x))

## Set the parameters for the method ############
 
# General parameters 
algo_options = {
    'q': 2,
    'p': 2,
    'sigma': 0.5,
    'disp_flag': 2,
    'memory_max_size': 0,
}

# Parameters for local phase
local_options = {
    'kappa': 0.75,
    'eps_thr': 10**(-4),
    'j_thr': np.inf,
    'act_thr': 0.95,
    'init_N_sample': 1,
    'norm_flag': 2,
    'sp_solver': 'IPOPT',
    'sp_solver_optns': {'tol': 10**(-15)}
}

algo_options['local_options'] = local_options

## Run method ########################

np.random.seed(0)

eps1 = 0.01
# x1 = np.zeros(n) 
# x1 = 4*np.random.rand(n)-2
x1 = np.array([-1.27376447e+00,  1.13168965e+00, -6.08025003e-01,  1.10455782e+00,
        2.36781359e-01, -3.27091271e-01, -1.65620658e+00, -6.98339692e-01,
       -1.09310674e-14,  7.22115710e-01, -6.70131321e-01, -9.35116274e-01,
        1.27704645e+00, -6.81415730e-02,  3.63159272e+00,  2.39550488e+00,
       -1.38588249e+00]) + 1e-3 * np.random.rand(n)

result_local_method = local_method(x1,eps1,problem_data,algo_options)

print('loss unreg: ',loss_unreg(result_local_method['best_x'],model))
print(repr(result_local_method['best_x']))

# from sys import exit
# exit()

## Plot optimization results ########################

# x_min = np.zeros(n)
# f_min = problem_data.oracle[0](x_min)

x_min = np.array([-1.27376447e+00,  1.13168965e+00, -6.08025003e-01,  1.10455782e+00,
        2.36781359e-01, -3.27091271e-01, -1.65620658e+00, -6.98339692e-01,
       -1.09310674e-14,  7.22115710e-01, -6.70131321e-01, -9.35116274e-01,
        1.27704645e+00, -6.81415730e-02,  3.63159272e+00,  2.39550488e+00,
       -1.38588249e+00])
f_min = problem_data.oracle[0](x_min)

# x_min = result_local_method['best_x']
# f_min = result_local_method['best_f_val']

x_arr = result_local_method['x_arr']
f_arr = result_local_method['f_arr']
eps_arr = result_local_method['eps_arr']
eval_counter_arr = result_local_method['eval_counter_arr']
numsample_arr = result_local_method['numsample_arr']
act_arr = result_local_method['act_arr']

size_x_arr = len(x_arr)
j_max = len(eps_arr)

lw = 1.5
ms = 10
# fig, (ax1, ax2, ax3, ax4, ax5,) = plt.subplots(2, 3)

plt.rcParams.update({
    "text.usetex": True
})

# (1,1) ###############################################

diff_list = [np.linalg.norm(x_arr[j] - x_min) for j in range(size_x_arr)]
filtered_list = [d for d in diff_list if d != 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d != 0]

ax1 = plt.subplot(2,3,1)
ax1.plot(filtered_inds,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)
ax1.plot(range(j_max),[np.log10(eps_arr[j]) for j in range(j_max)],'r.:',markersize=ms,linewidth=lw)

ax1.set_title("$\| x^j - x^* \|$")
ax1.grid()

# (1,2) ###############################################

diff_list = [val - f_min for val in f_arr]
filtered_list = [d for d in diff_list if d > 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d > 0]
filtered_evals = [eval_counter_arr[j][0] for j in filtered_inds]

ax2 = plt.subplot(2,3,2)
ax2.plot(filtered_evals,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)

ax2.set_title("$f(x^{j(l)}) - f(x^*)$")
ax2.grid()

# (1,3) ###############################################

ax3 = plt.subplot(2,3,3)
ax3.plot(range(j_max+1),np.log10(f_arr),'k.-',markersize=ms,linewidth=lw)

ax3.set_title("$f(x^j)$")
ax3.grid()

# (2,1) ###############################################

ax4 = plt.subplot(2,3,4)
ax4.plot(range(j_max),numsample_arr,'k.-',markersize=ms,linewidth=lw)

ax4.set_title("numsample")

# (2,2) ###############################################

ax5 = plt.subplot(2,3,5)
ax5.plot(range(j_max),act_arr,'k.-',markersize=ms,linewidth=lw)
ax5.set_ylim(-0.1,1.1)
ax5.grid()

ax5.set_title("activity")

plt.show()

## Visualize NN ########################

visualize(result_local_method['best_x'],model)