import cyipopt
import numpy as np

# Solving the subproblem via IPOPT
class HtbmSubproblem:

    def __init__(self,W,W_f,W_grad,W_hess,x,eps):
        
        self.W = W
        self.W_f = W_f
        self.W_grad = W_grad
        self.W_hess = W_hess
        self.x = x
        self.eps = eps
        self.k = len(W)
        self.n = W[0].shape[0]

    def objective(self, ztheta):
        """Returns the scalar value of the objective given x."""
        return ztheta[self.n]

    def gradient(self, ztheta):
        """Returns the gradient of the objective with respect to x."""
        return np.concatenate([np.zeros(self.n),[1]])

    def constraints(self, ztheta):
        """Returns the constraints."""
        c = np.zeros(self.k+1)

        for i in range(self.k):
            c[i] = ( 
                self.W_f[i] + 
                np.dot(self.W_grad[i],ztheta[:self.n] - self.W[i]) + 
                0.5*(np.matmul(ztheta[:self.n] - self.W[i],np.matmul(self.W_hess[i],ztheta[:self.n] - self.W[i])))
                - ztheta[self.n] 
            )

        c[self.k] = np.linalg.norm(ztheta[:self.n] - self.x)**2 - self.eps**2

        return c

    def jacobian(self, ztheta):
        """Returns the Jacobian of the constraints with respect to x."""
        J = np.zeros([self.k+1,self.n+1])

        for i in range(self.k):
            J[i,:self.n] = self.W_grad[i] + np.matmul(self.W_hess[i],ztheta[:self.n] - self.W[i])
            J[i,self.n] = -1

        J[self.k,:self.n] = 2*(ztheta[:self.n] - self.x)

        return J

    def jacobianstructure(self):
        return np.nonzero(np.ones([self.k+1,self.n+1]))

    def hessian(self, ztheta, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""

        H = np.zeros([self.n+1,self.n+1])
        for i in range(self.k):
            H[:self.n,:self.n] += lagrange[i]*self.W_hess[i]

        H[:self.n,:self.n] += lagrange[self.k]*2*np.eye(self.n)

        row, col = self.hessianstructure()
        return H[row, col]

    def hessianstructure(self):
        """Returns the row and column indices for non-zero values of the
        Hessian."""

        return np.nonzero(np.tril(np.ones([self.n+1,self.n+1])))

def solve(W,W_f,W_grad,W_hess,x,eps,sp_solver_options):

    k = len(W)
    n = W[0].shape[0]

    lb = np.concatenate([x - eps, [-np.inf]])
    ub = np.concatenate([x + eps, [np.inf]])

    cl = -np.inf*np.ones(k+1)
    cu = np.zeros(k+1)

    problem_obj = HtbmSubproblem(W,W_f,W_grad,W_hess,x,eps)

    x_mod = x.copy()
    x_mod[0] += 0.5*eps
    init_pt = np.concatenate([x_mod,[np.max(problem_obj.constraints(np.concatenate([x_mod,[0]]))) + 1]])

    nlp = cyipopt.Problem(
        n+1,
        m=len(cl),
        problem_obj=problem_obj,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    nlp.add_option('tol', sp_solver_options['tol'])
    nlp.add_option('constr_viol_tol', sp_solver_options['tol'])
    nlp.add_option('print_level', 0)
    nlp.add_option('line_search_method', 'cg-penalty')

    sol_IPOPT, info_IPOPT = nlp.solve(init_pt)

    return sol_IPOPT[:n], sol_IPOPT[n], info_IPOPT['mult_g'][-1]
