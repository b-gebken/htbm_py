import numpy as np
import time

from htbm_py.optimization_problem import OptimizationProblem

## Define problem ########################

### chained_CB3_I
# from test_functions.chained_CB3_I import chained_CB3_I
# n = 4
# problem_data = chained_CB3_I(n)

### simple_NN
from torch import nn

import numpy as np

from test_functions.simple_NN import simple_NN

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
problem_data = simple_NN(model)

n = 17

## Generate test points ########################

np.random.seed(0)

lb = -1*np.ones(n)
ub = 1*np.ones(n)

N_pts = 100
pts = lb + np.random.rand(N_pts,n) * (ub - lb)

## Check derivatives ########################

f = problem_data.oracle[0]
grad_f = problem_data.oracle[1]
hess_f = problem_data.oracle[2]

h_grad = 10**(-6)
grad_error_arr = np.zeros(N_pts)

h_hess = 10**(-5)
hess_error_arr = np.zeros(N_pts)

grad_time = 0
hess_time = 0

for i in range(N_pts):

    f_pts_i = f(pts[i])

    # Compute numerical gradient
    grad_i = np.zeros(n)
    for j in range(n):
        e_j = np.eye(1,n,j)[0]
        grad_i[j] = (f(pts[i] + h_grad*e_j) - f_pts_i)/h_grad

    start_eval = time.time()
    grad_exact = grad_f(pts[i])
    end_eval = time.time()
    grad_time += end_eval - start_eval

    grad_error_arr[i] = np.max(abs(grad_i - grad_exact))

    if grad_error_arr[i] > 10**(-5):
        print(grad_i)
        print(grad_f(pts[i]))

    # Compute numerical Hessian
    hess_i = np.zeros([n,n])
    for j in range(n):
        for k in range(j,n):
            e_j = np.eye(1,n,j)[0]
            e_k = np.eye(1,n,k)[0]
            hess_i[j,k] = (f(pts[i] + h_hess*e_j + h_hess*e_k) - f(pts[i] + h_hess*e_j - h_hess*e_k) - f(pts[i] - h_hess*e_j + h_hess*e_k) + f(pts[i] - h_hess*e_j - h_hess*e_k))/(4*h_hess**2)
            hess_i[k,j] = hess_i[j,k]

    start_eval = time.time()
    hess_exact = hess_f(pts[i])
    end_eval = time.time()
    hess_time += end_eval - start_eval

    hess_error_arr[i] = np.max(abs(hess_i - hess_exact))

    if hess_error_arr[i] > 10**(-1):
        print(hess_i)
        print(hess_f(pts[i]))

# print('Grad errors: ',grad_error_arr)
# print('Hess errors: ',hess_error_arr)
print('Max. grad error: ',np.max(grad_error_arr))
print('Max. hess error: ',np.max(hess_error_arr))
print('Avg. grad eval. time: ',grad_time/N_pts)
print('Avg. hess eval. time: ',hess_time/N_pts)