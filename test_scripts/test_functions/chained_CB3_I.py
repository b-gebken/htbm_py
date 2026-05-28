import numpy as np
from htbm_py.optimization_problem import OptimizationProblem

def chained_CB3_I(n):

    def f(x):
        val = 0
        for i in range(n-1):
            val += np.max([
                x[i]**4 + x[i+1]**2,
                (2 - x[i])**2 + (2 - x[i+1])**2,
                2*np.exp(-x[i] + x[i+1])
            ])

        return val
    
    def grad_f(x):
        grad = np.zeros(n)

        for i in range(n-1):
            I = np.argmax([
                x[i]**4 + x[i+1]**2,
                (2 - x[i])**2 + (2 - x[i+1])**2,
                2*np.exp(-x[i] + x[i+1])
            ])

            if I == 0:
                grad[i] += 4*x[i]**3
                grad[i+1] += 2*x[i+1]
            elif I == 1:
                grad[i] += -2*(2 - x[i])
                grad[i+1] += -2*(2 - x[i+1])
            else:
                grad[i] += -2*np.exp(-x[i] + x[i+1])
                grad[i+1] += 2*np.exp(-x[i] + x[i+1])


        return grad
    
    def hess_f(x):
        hess = np.zeros([n,n])

        for i in range(n-1):
            I = np.argmax([
                x[i]**4 + x[i+1]**2,
                (2 - x[i])**2 + (2 - x[i+1])**2,
                2*np.exp(-x[i] + x[i+1])
            ])

            if I == 0:
                hess[i,i] += 12*x[i]**2
                hess[i+1,i+1] += 2
            elif I == 1:
                hess[i,i] += 2
                hess[i+1,i+1] += 2
            else:
                hess[i,i] += 2*np.exp(-x[i] + x[i+1])
                hess[i,i+1] += -2*np.exp(-x[i] + x[i+1])
                hess[i+1,i] += -2*np.exp(-x[i] + x[i+1])
                hess[i+1,i+1] += 2*np.exp(-x[i] + x[i+1])

        return hess
    
    x0 = 2*np.ones(n)

    problem_data = OptimizationProblem(
        oracle=[f,grad_f,hess_f],
        x0=x0
    )
    
    return problem_data