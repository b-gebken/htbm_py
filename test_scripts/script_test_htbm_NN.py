# A script for training a neural network via HTBM
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from htbm_py.test_functions.loss_NN import loss_NN
from htbm_py.test_functions.loss_NN import visualize
from htbm_py.test_functions.loss_NN import loss_unreg

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

reg_param = 0.0001
N_data = 20
loss_fn_type = 'mse'
problem_data = loss_NN(model,reg_param,N_data,loss_fn_type)

n = problem_data.x0.shape[0]

## Set the parameters for the method #####
 
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
    'eps_thr': 1e-4,
    'j_thr': np.inf,
    'act_thr': 0.95,
    'init_N_sample': 1,
    'norm_flag': 2,
    'sp_solver': 'IPOPT',
    'sp_solver_optns': {'tol': 1e-15}
}

algo_options['local_options'] = local_options

## Initial data ##########################

# Approximated minimum, used to test local convergence
x_min = np.array([-1.2737644744674492e+00,  1.1316896534481780e+00,
       -6.0802500315902586e-01,  1.1045578177326596e+00,
        2.3678135927411392e-01, -3.2709127111716396e-01,
       -1.6562065817714948e+00, -6.9833969169552712e-01,
       0,  7.2211570940229886e-01,
       -6.7013132103765727e-01, -9.3511627427513966e-01,
        1.2770464483111723e+00, -6.8141573028809174e-02,
        3.6315927217865878e+00,  2.3955048787183748e+00,
       -1.3858824895083590e+00])

eps1 = 0.01
np.random.seed(0)
x1 = x_min + 1e-3 * np.random.rand(n)

## Run method ############################

result_local_method = local_method(x1,eps1,problem_data,algo_options)

print('Unregularized loss: ',loss_unreg(result_local_method['best_x'],model,N_data,loss_fn_type))

# np.set_printoptions(precision=20)
# print(repr(result_local_method['best_x']))

## Plot optimization results #############

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

plt.rcParams.update({
    "text.usetex": True
})

    # (1,1) ##############################

diff_list = [np.linalg.norm(x_arr[j] - x_min) for j in range(size_x_arr)]
filtered_list = [d for d in diff_list if d != 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d != 0]

ax1 = plt.subplot(2,3,1)
ax1.plot(filtered_inds,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)
ax1.plot(range(j_max),[np.log10(eps_arr[j]) for j in range(j_max)],'r.:',markersize=ms,linewidth=lw)

ax1.set_title("$\| x^j - x^* \|$")
ax1.grid()

    # (1,2) ##############################

diff_list = [val - f_min for val in f_arr]
filtered_list = [d for d in diff_list if d > 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d > 0]
filtered_evals = [eval_counter_arr[j][0] for j in filtered_inds]

ax2 = plt.subplot(2,3,2)
ax2.plot(filtered_evals,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)

ax2.set_title("$f(x^{j(l)}) - f(x^*)$")
ax2.grid()

    # (1,3) ##############################

ax3 = plt.subplot(2,3,3)
ax3.plot(range(j_max+1),np.log10(f_arr),'k.-',markersize=ms,linewidth=lw)

ax3.set_title("$f(x^j)$")
ax3.grid()

    # (2,1) ##############################

ax4 = plt.subplot(2,3,4)
ax4.plot(range(j_max),numsample_arr,'k.-',markersize=ms,linewidth=lw)
ax4.grid()

ax4.set_title("numsample")

    # (2,2) ##############################

ax5 = plt.subplot(2,3,5)
ax5.plot(range(j_max),act_arr,'k.-',markersize=ms,linewidth=lw)
ax5.set_ylim(-0.1,1.1)
ax5.grid()

ax5.set_title("activity")

    # (2,3) ##############################

ax6 = plt.subplot(2,3,6)
# ax6.plot(range(j_max+1),np.log10([(f_arr[i] - f_min)/(np.linalg.norm(x_arr[i] - x_min)**2) for i in range(j_max+1)]),'k.-',markersize=ms,linewidth=lw)
ax6.plot([np.log10(np.linalg.norm(x_arr[j] - x_min)) for j in range(j_max+1)],np.log10([(f_arr[j] - f_min)/(np.linalg.norm(x_arr[j] - x_min)**2) for j in range(j_max+1)]),'k.-',markersize=ms,linewidth=lw)
ax6.grid()

ax6.set_title("log10 growth param.")

plt.show()

## Visualize NN ##########################

visualize(result_local_method['best_x'],model,N_data)

