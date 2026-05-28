
import numpy as np
import htbm_py.solve_subproblem_IPOPT as solve_subproblem_IPOPT

n = 3
k = 2

W = [np.array([0.63,0.81]),np.array([-0.75,0.83]),np.array([0.26,-0.80])]
W_f = np.array([-0.44,0.09,0.92])
W_grad = [np.array([0.93,-0.68]),np.array([0.94,0.91]),np.array([-0.03,0.60])]
W_hess = [np.array([[-0.72,0.34],[0.34,0.58]]),
          np.array([[0.92,-0.31],[-0.31,0.70]]),
          np.array([[0.87,0.44],[0.44,0.49]])]
x = np.array([-0.22,0.31])
eps = 0.75

sp_solver_options = { 'tol' : 10**-10 }

z_bar, theta, mu = solve_subproblem_IPOPT.solve(W,W_f,W_grad,W_hess,x,eps,sp_solver_options)

print(z_bar)
print(theta)

