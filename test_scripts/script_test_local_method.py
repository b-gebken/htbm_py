import numpy as np
import matplotlib.pyplot as plt

from htbm_py.htbm import local_method
from htbm_py.optimization_problem import OptimizationProblem

## Define problem ########################

## chained_CB3_I
# from test_functions.chained_CB3_I import chained_CB3_I
# n = 5
# problem_data = chained_CB3_I(n)

## LW2019_85
from test_functions.lw2019_85 import lw2019_85
np.random.seed(0)
n = 8
k = 3
lambd = np.random.rand(k)
lambd = lambd / np.sum(lambd)
tmp = [2*np.random.rand(n) - 1 for _ in range(k)]
tmp2 = np.matmul(lambd,np.stack(tmp))
g_arr = [v - tmp2 for v in tmp]

H_arr = []
for _ in range(k):
    tmp = 2*np.random.rand(n,n) - 1
    H_arr.append(0.5 * (tmp + tmp.transpose()))

c_arr = [np.random.rand() for _ in range(k)]

problem_data = lw2019_85(g_arr,H_arr,c_arr)

## half_and_half
# from test_functions.half_and_half import half_and_half
# problem_data = half_and_half()

## Set the parameters for the method ############
 
# General parameters 
algo_options = {
    'q': 2,
    'p': 2,
    'sigma': 0.5,
    'disp_flag': 1,
    'memory_max_size': 10000,
}

# Parameters for local phase
local_options = {
    'kappa': 0.75,
    'eps_thr': 10**(-3),
    'j_thr': np.inf,
    'act_thr': 0.95,
    'init_N_sample': 1,
    'norm_flag': 2,
    'sp_solver': 'IPOPT',
    'sp_solver_optns': {'tol': 10**(-10)}
}

algo_options['local_options'] = local_options

## Run method ########################

eps1 = 10
x1 = np.ones(n)

result_local_method = local_method(x1,eps1,problem_data,algo_options)

## Plots ########################

x_min = np.zeros(n)
f_min = problem_data.oracle[0](x_min)

# x_min = result_local_method['best_x']
# f_min = result_local_method['best_f_val']

x_arr = result_local_method['x_arr']
f_arr = result_local_method['f_arr']
eps_arr = result_local_method['eps_arr']
eval_counter_arr = result_local_method['eval_counter_arr']

size_x_arr = len(x_arr)
j_max = len(eps_arr)

lw = 1.5
ms = 10
fig, (ax1, ax2) = plt.subplots(1, 2)

plt.rcParams.update({
    "text.usetex": True
})

# (1,1) ###############################################

diff_list = [np.linalg.norm(x_arr[j] - x_min) for j in range(size_x_arr)]
filtered_list = [d for d in diff_list if d != 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d != 0]

ax1.plot(filtered_inds,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)
ax1.plot(range(j_max),[np.log10(eps_arr[j]) for j in range(j_max)],'r.:',markersize=ms,linewidth=lw)

ax1.set_title("$\| x^j - x^* \|$")
ax1.grid()

# (1,2) ###############################################

diff_list = [val - f_min for val in f_arr]
filtered_list = [d for d in diff_list if d > 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d > 0]
filtered_evals = [eval_counter_arr[j][0] for j in filtered_inds]

ax2.plot(filtered_evals,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)

ax2.set_title("$f(x^{j(l)}) - f(x^*)$")
ax2.grid()

plt.show()
