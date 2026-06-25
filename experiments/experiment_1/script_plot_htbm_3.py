# A script for training a neural network via HTBM
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from htbm_py.test_functions.loss_NN import loss_NN
from htbm_py.test_functions.loss_NN import visualize
from htbm_py.test_functions.loss_NN import loss_unreg

from htbm_py.htbm import local_method

SAVE_PLOT = False

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
N_data = 100
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
    'eps_thr': 1e-6,
    'j_thr': np.inf,
    'act_thr': 0.95,
    'init_N_sample': 1,
    'norm_flag': 2,
    'sp_solver': 'IPOPT',
    'sp_solver_optns': {'tol': 1e-15}
}

algo_options['local_options'] = local_options

## Initial data ##########################

# Approximated minimum for N_data = 20, reg_param = 0
x_min = np.array([ 1.5516524323086034 ,  1.048175016093194  , -0.7381008874112601 ,
       -0.8022944455194168 ,  0.14897712188505186,  0.22579360496294085,
       -2.194995397652173  ,  0.6885995431383544 , -0.8986436125804929 ,
        1.1783570851791754 , -0.41464331894467754, -0.8550271274126808 ,
        0.48182454692832233,  0.7498286832597288 , -2.6475327180665937 ,
       -1.8012898448227466 ,  1.8864978908494021 ])

np.random.seed(0)
x1 = x_min + 1*1e-3 * (2*np.random.rand(n)-1)
eps1 = 1*1e-2

## Run method ############################

result_local_method = local_method(x1,eps1,problem_data,algo_options)

print('Unregularized loss: ',loss_unreg(result_local_method['best_x'],model,N_data,loss_fn_type))
print('Norm of gradient:   ',np.linalg.norm(problem_data.oracle[1](result_local_method['best_x'])))

## Plot optimization results #############

# f_min = problem_data.oracle[0](x_min)

x_min = result_local_method['best_x']
f_min = result_local_method['best_f_val']

x_arr = result_local_method['x_arr']
f_arr = result_local_method['f_arr']
eps_arr = result_local_method['eps_arr']
eval_counter_arr = result_local_method['eval_counter_arr']
numsample_arr = result_local_method['numsample_arr']
act_arr = result_local_method['act_arr']

if result_local_method['best_f_val'] < f_min:
    print('Better point than x_min found! Diff = ', f_min - result_local_method['best_f_val'])
    x_min = result_local_method['best_x']
    f_min = result_local_method['best_f_val']
    np.set_printoptions(precision=20)
    print(repr(result_local_method['best_x']))

size_x_arr = len(x_arr)
j_max = len(eps_arr)

lw = 1.5
ms = 10

plt.rcParams.update({
    "text.usetex": True
})
plt.rcParams['font.size'] = 15

    # (1,1) ##############################

diff_list = [np.linalg.norm(x_arr[j] - x_min) for j in range(size_x_arr)]
filtered_list = [d for d in diff_list if d != 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d != 0]

ax1 = plt.subplot(2,3,1)
ax1.plot(filtered_inds,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)
ax1.plot(range(j_max),[np.log10(eps_arr[j]) for j in range(j_max)],'r.:',markersize=ms,linewidth=lw)
[ax1.axvline(x = j, linestyle = '--', color = 'k') for j, d in enumerate(diff_list) if d == 0]

ax1.set_title("$\log_{10}(\| x^j - x^{best} \|)$")
ax1.grid()
ax1.set_xlabel('$j$')
ax1.set_yticks(np.arange(np.floor(np.min(np.log10(np.array(filtered_list)))),np.ceil(np.max(np.log10(np.array(filtered_list))))+1,step=2))
ax1.set_ylim(np.floor(np.min(np.log10(np.array(filtered_list))))-0.25,np.ceil(np.max(np.log10(np.array(filtered_list))))+0.25)

    # (1,2) ##############################

diff_list = [val - f_min for val in f_arr]
filtered_list = [d for d in diff_list if d > 0]
filtered_inds = [j for j, d in enumerate(diff_list) if d > 0]
filtered_evals = [eval_counter_arr[j][0] for j in filtered_inds]

ax2 = plt.subplot(2,3,2)
ax2.plot(filtered_evals,np.log10(np.array(filtered_list)),'k.-',markersize=ms,linewidth=lw)
[ax2.axvline(x = eval_counter_arr[j][0], linestyle = '--', color = 'k') for j, d in enumerate(diff_list) if d == 0]

ax2.set_title("$\log_{10}(f(x^{j(l)}) - f(x^{best}))$")
ax2.grid()
ax2.set_xlabel('Oracle calls')

    # (1,3) ##############################

ax3 = plt.subplot(2,3,3)
visualize(result_local_method['best_x'],model,N_data,axes = ax3, fmt = 'r--')
ax3.grid()
ax3.set_title("Model and data")

    # (2,1) ##############################

ax4 = plt.subplot(2,3,4)
ax4.plot(range(j_max),numsample_arr,'k.-',markersize=ms,linewidth=lw)
ax4.grid()

ax4.set_title("No. of samples")
ax4.set_xlabel('$j$')
ax4.set_yticks(np.arange(0,np.ceil(np.max(numsample_arr))+1,step=1))
ax4.set_ylim(0-0.25,np.ceil(np.max(numsample_arr))+0.25)
ax4.axhline(y = 1, linestyle = '--', color = 'r')

    # (2,2) ##############################

ax5 = plt.subplot(2,3,5)
ax5.plot(range(j_max),act_arr,'k.-',markersize=ms,linewidth=lw)
ax5.set_ylim(-0.1,1.1)
ax5.grid()

ax5.set_title("TR activity")
ax5.set_xlabel('$j$')

    # (2,3) ##############################

filtered_norms = [np.log10(np.linalg.norm(x_arr[j] - x_min)) for j in filtered_inds]
filtered_grparam = [np.log10((f_arr[j] - f_min)/(np.linalg.norm(x_arr[j] - x_min)**2)) for j in filtered_inds]

ax6 = plt.subplot(2,3,6)
ax6.plot(filtered_norms,filtered_grparam,'k.',markersize=ms,linewidth=lw)
ax6.grid()

ax6.set_title("$\log_{10}($quad. gr. param.$)$")
ax6.set_xlabel('$\log_{10}(\| x^j - x^{\mathrm{fin}} \|)$')
ax6.set_xticks(np.arange(np.floor(np.min(filtered_norms)),np.ceil(np.max(filtered_norms))+1,step=1))
ax6.set_yticks(np.arange(np.floor(np.min(filtered_grparam)),np.ceil(np.max(filtered_grparam))+1,step=3))
ax6.set_xlim(np.floor(np.min(filtered_norms))-0.25,np.ceil(np.max(filtered_norms))+0.25)
ax6.set_ylim(np.floor(np.min(filtered_grparam))-0.25,np.ceil(np.max(filtered_grparam))+0.25)

fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.tight_layout()
fig.subplots_adjust(top=0.85)
fig.text(
    0.01, 0.92,
    f"N_data: {N_data}, "
    f"Reg. param: {reg_param:.2e}, "
    f"Loss type: {loss_fn_type}, "
    f"$f(x^1)$: {problem_data.oracle[0](x1):.2e}, "
    f"Unreg. loss at $x^1$: {loss_unreg(x1,model,N_data,loss_fn_type):.2e}, "
    "\n"
    f"$f(x^{{best}})$: {result_local_method['best_f_val']:.2e}, "
    f"$f(x^1) - f(x^{{best}})$: {problem_data.oracle[0](x1)-result_local_method['best_f_val']:.2e}, "
    f"$\\| \\nabla f(x^{{best}}) \\|$: {np.linalg.norm(problem_data.oracle[1](result_local_method['best_x'])):.2e}, "
    f"Unreg. loss at $x^{{best}}$: {loss_unreg(result_local_method['best_x'],model,N_data,loss_fn_type):.2e}",
    fontsize=15,
    va='bottom'
)

if SAVE_PLOT:
    plt.savefig('experiments/experiment_1/plot_3.png',bbox_inches='tight',dpi=300)
else:
    plt.show()