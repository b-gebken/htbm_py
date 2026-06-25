# A script for testing the function descent_direction from htbm_py.dgs
import numpy as np

from htbm_py.dgs import descent_direction
from htbm_py.memory import Memory

import matplotlib.pyplot as plt

SAVE_PLOT = False

## loss_NN
from torch import nn
from htbm_py.test_functions.loss_NN import loss_NN

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

reg_param = 0
N_data = 7
loss_fn_type = 'mae'
problem_data = loss_NN(model,reg_param,N_data,loss_fn_type)

## Run method ############################

np.random.seed(0)

x = np.array([-1.3680836250807094e+00, -5.9759500488706652e-01,
        1.6947460520251982e-03,  4.9334030259655570e-01,
        6.5574534809355634e-01,  4.0165897886643388e-03,
       -1.1976560224716530e+00,  1.2503824733091247e+00,
        1.8700600915289446e-03, -6.7247227572259016e-01,
       -1.0490239672885917e-02, -7.2849569052055069e-04,
       -1.3689648353097960e-03,  1.0948374461809136e+00,
       -2.4403545542714453e+00, -1.7976421896544210e+00,
        1.9150885074842388e+00])
f_x = problem_data.oracle[0](x)
f = problem_data.oracle[0]
subgrad_f = problem_data.oracle[1]
eps = 1e-8
delta = 1e-7
c = 0.9
rand_sample_N = 0
memory = Memory(1000)
eval_counter = np.zeros(2)
disp_flag = 3

v, f_eps_v, num_sample = descent_direction(x,f_x,f,subgrad_f,eps,delta,c,rand_sample_N,memory,eval_counter,disp_flag)

print('||v|| = ',np.linalg.norm(v,ord=2))
print('eval_counter = ',eval_counter)

## Computing orthogonal space of subgradients
# grad_mat = np.asarray(memory.oracle_vals)
# n = x.shape[0]
# U, S, Vh = np.linalg.svd(grad_mat, full_matrices=True)
# print('Singular values:',S)
# null_sv = sum(S <= 1e-10)
# print('Dimension of null space:',n-grad_mat.shape[0]+null_sv)
# null_space = Vh.transpose()[:,-(n-grad_mat.shape[0]+null_sv):]
# h = 1e-5
# print('Max. of dir. deriv. in null directions:',max([(f(x + h * null_space[:,i]) - f(x))/h for i in range(null_space.shape[1])]))

## Plotting the subgradients
# [plt.plot(memory.oracle_vals[i]) for i in range(len(memory.sample_pts))]
# plt.show()

## Checking lower-C^1 structure (cf. [RW1998], Thm. 10.31)
h = 1e-5
plt.rcParams.update({
    "text.usetex": True
})
plt.rcParams['font.size'] = 15

        # (1,1) ##############################

dir_deriv_list = [(f(x + h * g) - f(x))/h for g in memory.oracle_vals]
max_dot_list = [np.max([np.dot(grad,g) for grad in memory.oracle_vals]) for g in memory.oracle_vals]

print(f"Max. violation: {np.max(np.asarray(max_dot_list) - np.asarray(dir_deriv_list))}")

ax1 = plt.subplot(2,2,1)

ax1.plot(dir_deriv_list,'r-')
ax1.plot(max_dot_list,'b-')

ax1.legend(['$df(x,g)$','$\max_{{\\xi \\in \\partial f(x)}} \\langle \\xi,g \\rangle$'])

ax1.set_xlabel('Sampled gradients')
ax1.grid()
ax1.set_title(f"Max. violation: {np.max(np.asarray(max_dot_list) - np.asarray(dir_deriv_list)):.4e}")

        # (2,1) ##############################

U, S, Vh = np.linalg.svd(np.asarray(memory.oracle_vals), full_matrices=True)

ax3 = plt.subplot(2,2,3)
ax3.plot(np.log10(S),'k.-')
ax3.grid()
ax3.set_title("$\log_{10}($Singular values$)$")

        # (1,2) ##############################

v, f_eps_v, num_sample = descent_direction(x,f_x,f,subgrad_f,eps,delta,c,100,memory,eval_counter,disp_flag)

dir_deriv_list = [(f(x + h * g) - f(x))/h for g in memory.oracle_vals]
max_dot_list = [np.max([np.dot(grad,g) for grad in memory.oracle_vals]) for g in memory.oracle_vals]

print(f"Max. violation: {np.max(np.asarray(max_dot_list) - np.asarray(dir_deriv_list))}")

ax2 = plt.subplot(2,2,2)

ax2.plot(dir_deriv_list,'r-')
ax2.plot(max_dot_list,'b-')

ax2.legend(['$df(x,g)$','$\max_{{\\xi \\in \\partial f(x)}} \\langle \\xi,g \\rangle$'])

ax2.set_xlabel('Sampled gradients')
ax2.grid()
ax2.set_title(f"Max. violation: {np.max(np.asarray(max_dot_list) - np.asarray(dir_deriv_list)):.4e}")

        # (2,2) ##############################

U, S, Vh = np.linalg.svd(np.asarray(memory.oracle_vals), full_matrices=True)

ax3 = plt.subplot(2,2,4)
ax3.plot(np.log10(S),'k.-')
ax3.grid()
ax3.set_title("$\log_{10}($Singular values$)$")

fig = plt.gcf()
fig.set_size_inches(10, 8)

plt.tight_layout()

if SAVE_PLOT:
    plt.savefig('experiments/experiment_2/plot_2_2.png',bbox_inches='tight',dpi=300)
else:
    plt.show()