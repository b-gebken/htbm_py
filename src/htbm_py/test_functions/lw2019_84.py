# Generates an oracle for the test function (8.4) from [Lewis, Wylie (2019), "A
# simple Newton method for local nonsmooth optimization"].

import numpy as np
from htbm_py.optimization_problem import OptimizationProblem

def lw2019_84(g_arr,H_arr,c_arr):

    def f(x):
        k = len(H_arr)
        tmp = np.zeros(k)
        for i in range(k):
            tmp[i] = np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4

        return np.max(tmp)
    
    def grad_f(x):
        k = len(H_arr)
        tmp = np.zeros(k)
        for i in range(k):
            tmp[i] = np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4

        I = np.argmax(tmp)

        return g_arr[I] + H_arr[I] @ x + c_arr[I]/24 * 4*np.linalg.norm(x)**2 * x
    
    def hess_f(x):
        k = len(H_arr)
        tmp = np.zeros(k)
        for i in range(k):
            tmp[i] = np.dot(g_arr[i],x) + 0.5 * np.dot(x,np.matmul(H_arr[i],x)) + c_arr[i]/24 * np.linalg.norm(x)**4

        I = np.argmax(tmp)

        return H_arr[I] + c_arr[I]/24 * (8*np.outer(x,x) + np.diag(4*np.linalg.norm(x)**2 * np.ones(n)))
    
    k = len(g_arr)
    n = g_arr[0].shape[0]
    x0 = np.ones(n)

    problem_data = OptimizationProblem(
        oracle=[f,grad_f,hess_f],
        x0=x0
    )
    
    return problem_data