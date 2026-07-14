# A script for training a neural network via DGS

import numpy as np
import matplotlib.pyplot as plt

from htbm_py.dgs import dgs

## Define problem ########################

### LW2019_85
# np.random.seed(0)
# from htbm_py.test_functions.lw2019_85 import lw2019_85
# n = 10
# k = 6
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

# np.random.seed(None)
# problem_data.x0 = np.zeros(n) + 2*np.random.rand(n) - 1

# def lw2019_85_act(x,g_arr,H_arr,c_arr):
#     s = np.zeros(k)
#     act = 0
#     for i in range(k):
#         s[i] = np.sign(np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4)
#         if s[i] == 1:
#             act += 2**i
#         elif s[i] == 0:
#             print('Warning: sign is zero!')

#     return act

### LW2019_84
np.random.seed(0)
from htbm_py.test_functions.lw2019_84 import lw2019_84
n = 10
k = 6
lambd = np.random.rand(k)
lambd = lambd / np.sum(lambd)
tmp = [2*np.random.rand(n) - 1 for _ in range(k)]
tmp2 = np.matmul(lambd,np.stack(tmp))
g_arr = [v - tmp2 for v in tmp]

H_arr = []
for _ in range(k):
    tmp = 2*np.random.rand(n,n) - 1
    H_arr.append(tmp @ tmp.transpose())

c_arr = [np.random.rand() for _ in range(k)]

problem_data = lw2019_84(g_arr,H_arr,c_arr)

np.random.seed(2)
problem_data.x0 = np.zeros(n) + 2*np.random.rand(n) - 1

def lw2019_84_vec(x,g_arr,H_arr,c_arr):
    k = len(H_arr)
    tmp = np.zeros(k)
    for i in range(k):
        tmp[i] = np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4

    return tmp

def lw2019_84_act(x,g_arr,H_arr,c_arr):
    return np.argmax(lw2019_84_vec(x,g_arr,H_arr,c_arr))

def lw2019_84_act_thresh(x,g_arr,H_arr,c_arr,thresh):
    tmp = lw2019_84_vec(x,g_arr,H_arr,c_arr)
    return sum(np.max(tmp) - tmp < thresh)

## Set the parameters for the method #####

j_max = 1
kappa_eps = 0.1
eps0 = 1

algo_options = {
    #'eps_arr': eps0*kappa_eps**(np.arange(j_max)), # different to Matlab because index offset
    'eps_arr': np.array([1e-2]),
    'delta_arr': 1e-5 * np.ones(j_max),
    'rand_sample_N': 50,
    'memory_max_size': 0,
    'c': 0.5,
    'ls_flag': 'eps',
    'max_iter': 10000,
    'disp_flag': 2
}

## Run DGS ###############################

x_opt, f_opt, x_arr_list, fx_arr_list, eval_counter, num_sample_arr_list = dgs(problem_data,algo_options)

## Plots #################################

x_min = np.zeros(n)
f_min = 0

numiter_arr = [len(fx_arr) for fx_arr in fx_arr_list]
num_total_iter = sum(numiter_arr)

x_all = []
fx_all = np.zeros(num_total_iter)
num_sample_all = np.zeros(num_total_iter)
counter = 0
for j in range(j_max):
    x_all.extend(x_arr_list[j])
    fx_all[counter:counter+len(x_arr_list[j])] = np.asarray(fx_arr_list[j])
    num_sample_all[counter:counter+len(x_arr_list[j])] = np.asarray(num_sample_arr_list[j])
    counter = counter + len(x_arr_list[j])

    # (1,1) ##############################

ax1 = plt.subplot(2,3,1)
ax1.plot([np.log10(np.linalg.norm(x - x_min)) for x in x_all],'k.-')
ax1.grid()

ax1.set_title('x error')

    # (1,2) ##############################

ax2 = plt.subplot(2,3,2)
ax2.plot(np.log10(fx_all - f_min),'k.-')
ax2.grid()

ax2.set_title('f error')

    # (1,3) ##############################

ax3 = plt.subplot(2,3,3)
ax3.plot(numiter_arr,'k.-')
ax3.grid()

ax3.set_title('#Iter')

    # (2,1) ##############################

ax4 = plt.subplot(2,3,4)
ax4.plot(num_sample_all,'k.-')
ax4.grid()

ax4.set_title('#Samples')

    # (2,2) ##############################

ax5 = plt.subplot(2,3,5)
ax5.plot([np.dot(x_all[i+2] - x_all[i+1],x_all[i+1] - x_all[i])/(np.linalg.norm(x_all[i+2] - x_all[i+1])*np.linalg.norm(x_all[i+1] - x_all[i])) for i in range(len(x_all)-2)],'k.-')
ax5.grid()

ax5.set_title('Angle')

    # (2,3) ##############################

ax6 = plt.subplot(2,3,6)

# ax6.plot([lw2019_84_act_thresh(x,g_arr,H_arr,c_arr,1e-3) for x in x_all])
# ax6.axhline(y = k, linestyle = '--', color = 'r')
# ax6.grid()

vec_list = [lw2019_84_vec(x,g_arr,H_arr,c_arr) for x in x_all]
vec_list = [vec_list[j] - np.max(vec_list[j]) for j in range(len(vec_list))]
for i in range(k):
    ax6.plot([vec_list[j][i] for j in range(len(vec_list))],'.-')
ax6.grid()

ax6.set_title('Activity')


plt.show()