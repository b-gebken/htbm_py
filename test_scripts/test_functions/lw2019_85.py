# Generates an oracle for the test function (8.5) from [Lewis, Wylie (2019), "A
# simple Newton method for local nonsmooth optimization"].

import numpy as np
from htbm_py.optimization_problem import OptimizationProblem

def lw2019_85(g_arr,H_arr,c_arr):

    def f(x):
        val = 0
        for i in range(k):
            val += np.abs(np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4)

        return val
    
    def grad_f(x):
        grad = np.zeros(n)

        for i in range(k):
            s = np.sign(np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4)
            if s == 0:
                s = 1

            grad += s * (g_arr[i] + np.matmul(H_arr[i],x) + c_arr[i]/24 * 4*np.linalg.norm(x)**2 * x)

        return grad
    
    def hess_f(x):
        hess = np.zeros([n,n])

        for i in range(k):
            s = np.sign(np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4)
            if s == 0:
                s = 1

            hess += s * (H_arr[i] + c_arr[i]/24 * (8*np.outer(x,x) + np.diag(4*np.linalg.norm(x)**2 * np.ones(n))))


        return hess
    
    k = len(g_arr)
    n = g_arr[0].shape[0]
    x0 = np.ones(n)

    problem_data = OptimizationProblem(
        oracle=[f,grad_f,hess_f],
        x0=x0
    )
    
    return problem_data