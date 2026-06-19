# A script for testing the function descent_direction from htbm_py.dgs

import numpy as np

from htbm_py.dgs import descent_direction
from htbm_py.memory import Memory

## Define problem ########################

## chained_CB3_I
# from htbm_py.test_functions.chained_CB3_I import chained_CB3_I
# n = 5
# problem_data = chained_CB3_I(n)

## LW2019_85
from htbm_py.test_functions.lw2019_85 import lw2019_85
np.random.seed(0)
n = 10
k = 5
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

## Run method ############################

x = 0*np.ones(n)
f_x = problem_data.oracle[0](x)
f = problem_data.oracle[0]
subgrad_f = problem_data.oracle[1]
eps = 0.005
delta = 1e-5
c = 0.9
rand_sample_N = 1
memory = Memory(1000)
eval_counter = np.zeros(2)
disp_flag = 3

v, f_eps_v, num_sample = descent_direction(x,f_x,f,subgrad_f,eps,delta,c,rand_sample_N,memory,eval_counter,disp_flag)

print('||v|| = ',np.linalg.norm(v,ord=2))
print('eval_counter = ',eval_counter)