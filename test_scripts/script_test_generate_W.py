# A script for testing the bundling mechanism in the HTBM

import numpy as np

from htbm_py.htbm import generate_W
from htbm_py.memory import Memory

## Define problem ########################

## chained_CB3_I
from htbm_py.test_functions.chained_CB3_I import chained_CB3_I
n = 5
problem_data = chained_CB3_I(n)

## LW2019_85
# np.random.seed(0)
# from htbm_py.test_functions.lw2019_85 import lw2019_85
# n = 5
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

## half and half
# from htbm_py.test_functions.half_and_half import half_and_half
# n = 8
# problem_data = half_and_half()

## Run method ########################
# x = 2*np.random.rand(n) - 1
x = np.ones(n)

algo_options = {
    'q': 2,
    'p': 2,
    'sigma': 0.5,
    'disp_flag': 2,
    'memory_max_size': 10000,
    'kappa': 0.75,
    'eps_thr': 1e-3,
    'j_thr': np.inf,
    'act_thr': 0.95,
    'init_N_sample': 1,
    'norm_flag': 2,
    'sp_solver': 'IPOPT',
    'sp_solver_optns': {'tol': 1e-10}
}

mem = Memory(algo_options['memory_max_size'])
eval_counter = np.zeros(algo_options['q']+1)

z_bar, f_z_bar, mu, numsample = generate_W(
    x=x,
    eps=0.1,
    f_x=problem_data.oracle[0](x),
    c=np.inf,
    reusing_eps_tolerance=1e-7,
    memory=mem,
    eval_counter=eval_counter,
    problem_data=problem_data,
    algo_options=algo_options,
    sp_options=algo_options
    )
