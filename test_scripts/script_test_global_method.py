# A script for testing HTBM on classical test functions 

import numpy as np
import matplotlib.pyplot as plt

from htbm_py.htbm import global_method

## Define problem ########################

## chained_CB3_I
from htbm_py.test_functions.chained_CB3_I import chained_CB3_I
n = 5
problem_data = chained_CB3_I(n)

## LW2019_85
# from htbm_py.test_functions.lw2019_85 import lw2019_85
# np.random.seed(0)
# n = 8
# k = 3
# lambd = np.random.rand(k)
# lambd = lambd / np.sum(lambd)
# tmp = [2*np.random.rand(n) - 1 for _ in range(k)]
# tmp2 = np.matmul(lambd,np.stack(tmp))
# g_arr = [v - tmp2 for v in tmp]

# H_arr = []
# for _ in range(k):
#     tmp = 2*np.random.rand(n,n) - 1
#     H_arr.append(0.5 * (tmp + tmp.transpose()))

# c_arr = [np.random.rand() for _ in range(k)]

# problem_data = lw2019_85(g_arr,H_arr,c_arr)

## half_and_half
# from htbm_py.test_functions.half_and_half import half_and_half
# problem_data = half_and_half()

## Set the parameters for the method #####
 
# General parameters 
algo_options = {
    'q': 2,
    'p': 2,
    'sigma': 0.5,
    'disp_flag': 2,
    'memory_max_size': 100,
    'local_flag': False
}

# Parameters for global phase
global_options = {
    'delta_arr': 10.0**np.arange(1,-5,-1),
    'tau_arr': 1e-5 * np.ones(6),
    'c': 0.1,
    'i_max': 10000,
    'init_N_sample': 1,
    'norm_flag': 2,
    'sp_solver': 'IPOPT',
    'sp_solver_optns': {'tol': 1e-10}
}

algo_options['global_options'] = global_options

# Parameters for local phase
local_options = {
    'kappa': 0.75,
    'eps_thr': 1e-3,
    'j_thr': 2, #np.inf,
    'act_thr': 0.95,
    'init_N_sample': 1,
    'norm_flag': 2,
    'sp_solver': 'IPOPT',
    'sp_solver_optns': {'tol': 1e-10}
}

algo_options['local_options'] = local_options

## Run method ############################

result_global_method = global_method(problem_data,algo_options)

## Plots #################################

n = problem_data.x0.shape[0]

x_min = np.ones(n)
f_min = problem_data.oracle[0](x_min)

x_cell = result_global_method['x_cell']
x_all = []
for j in range(len(x_cell)):
    x_all.extend(x_cell[j])

lw = 1.5
ms = 10

plt.plot([np.log10(np.linalg.norm(x_all[i] - x_min,ord=2)) for i in range(len(x_all))],'k.-')
plt.grid()
plt.show()

