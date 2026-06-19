# Generates an oracle for the test function half-and-half from Section 5.5 in
# [Lewis, Overton (2008), "Nonsmooth optimization via BFGS"]

import numpy as np
from htbm_py.optimization_problem import OptimizationProblem

def half_and_half():

    def f(x):
        return np.sqrt(np.matmul(x,np.matmul(A,x))) + np.matmul(x,np.matmul(B,x))
    
    def grad_f(x):
        return np.matmul(x,np.matmul(A,x))**(-1/2) * np.matmul(A,x) + 2*np.matmul(B,x)
    
    def hess_f(x):
        return -np.matmul(x,np.matmul(A,x))**(-3/2) * np.outer(np.matmul(A,x),np.matmul(A,x)) + np.matmul(x,np.matmul(A,x))**(-1/2) * A + 2*B
    
    n = 8
    A = np.diag([1,0,1,0,1,0,1,0])
    B = np.diag([1/(j**2) for j in range(1,9)])

    x0 = 20.08 * np.ones(8)

    problem_data = OptimizationProblem(
        oracle=[f,grad_f,hess_f],
        x0=x0
    )
    
    return problem_data