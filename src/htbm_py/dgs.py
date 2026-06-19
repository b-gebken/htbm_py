import numpy as np
import time

from numpy.linalg import norm

from htbm_py.htbm import sample_hypersphere
from htbm_py.memory import Memory

from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-15
solvers.options['reltol'] = 1e-15
solvers.options['feastol'] = 1e-15

def smallest_norm_element(grad_list):
    """Compute element with the smallest norm in convex hull of grad_list
    """

    num_grad = len(grad_list)

    grad_mat = np.asarray(grad_list).transpose()

    P = matrix(grad_mat.transpose() @ grad_mat)
    q = matrix(np.zeros(num_grad))
    G = matrix(-np.eye(num_grad))
    h = matrix(np.zeros(num_grad))
    A = matrix(np.ones([1,num_grad]))
    b = matrix([1.0])
    sol=solvers.qp(P, q, G, h, A, b)

    return np.squeeze(grad_mat @ np.array(sol['x']))

def descent_direction(x,f_x,f,subgrad_f,eps,delta,c,rand_sample_N,memory,eval_counter,disp_flag):
    """Compute descent direction
    
    Algorithm 2 in [GP2021] (or Algorithm 4.2 in [G2022]) combined with
    Algorithm 2 in [G2024b]. (See function dgs for the reference list.)
    """

    n = x.shape[0]

    # For computing num_sample in the end
    eval_counter_old = eval_counter.copy()

    # Step 1 in [GP2021] (Deterministic initial approximation)
    sample_pts = [x]
    W = [subgrad_f(x)]; eval_counter[1] += 1

    # Random initial approximation
    for k in range(1,rand_sample_N):
        sample_pts.append(sample_hypersphere(n))
        W.append(subgrad_f(sample_pts[k])); eval_counter[1] += 1

    if memory.max_size > 0:
        memory.add(sample_pts,W)

    # Add subgradients at sample points in B_eps(x) from memory
    reusing_eps_tolerance = 1e-7
    if (memory.max_size > 0) and (len(memory.sample_pts) > 0):
        inds = norm(np.stack(memory.sample_pts) - x,axis=1,ord=2) <= eps + reusing_eps_tolerance
    
        sample_pts_mem = [y for y, inds in zip(memory.sample_pts,inds) if inds]
        W_mem = [g for g, inds in zip(memory.oracle_vals,inds) if inds]

        sample_pts.extend(sample_pts_mem)
        W.extend(W_mem)
    
    while True:
        # Step 2 in [GP2021] (Compute direction based on W)
        v = -smallest_norm_element(W)

        # Step 3 in [GP2021] (Stopping criterion)
        if norm(v,ord=2) <= delta:
            f_eps_v = np.nan
            
            if disp_flag >= 3:
                print('                ...found v with ||v|| = ',norm(v,ord=2),' for |W| = ',len(W))
                print('                Smallest gradient norm: ',np.min([norm(xi,ord=2) for xi in W]))

            break

        # Step 4 in [GP2021] (Check for sufficient decrease)
        f_eps_v = f(x + eps/norm(v,ord=2) * v); eval_counter[0] += 1
        if f_eps_v - f_x > -c*eps*norm(v,ord=2):
            # Apply Algorithm 2 from [G2024b]
            c_min = -(f_eps_v - f_x)/(eps*norm(v,ord=2))
            c_tilde = (c + c_min)/2

            # Step 1 in [G2024b] (Initialization)
            a = 0
            b = eps/norm(v,ord=2)
            t = (a+b)/2
            bis_flag = 0 # 0 - Start; 1 - Right; 2 - Left

            while True:
                # Step 2 in [G2024b] (Evaluate subgradient)
                xi = subgrad_f(x + t*v); eval_counter[1] += 1

                # Step 3 in [G2024b] (Check if xi is in conv(W) and if not, add it to W and stop)
                if np.dot(xi,v) > -c*norm(v,ord=2)**2:
                    sample_pts.append(x + t*v)
                    W.append(xi)

                    if memory.max_size:
                        memory.add([x + t*v], [xi])

                    break

                # Step 4 in [G2024b] (Perform bisection of [a,b])
                if bis_flag == 2:
                    h_b = h_t
                elif bis_flag == 0:
                    h_b = f_eps_v - f_x + c_tilde*b*norm(v,ord=2)**2

                h_t = f(x + t*v) - f_x + c_tilde*t*norm(v,ord=2)**2; eval_counter[0] += 1

                if h_b > h_t:
                    a = t
                    bis_flag = 1
                else:
                    b = t
                    bis_flag = 2

                # Step 5 in [G2024b] (Update t)
                t = (a+b)/2
        else:
            break

    num_sample = eval_counter[1] - eval_counter_old[1]

    return v, f_eps_v, num_sample

def dgs(problem_data,algo_options):
    """Deterministic gradient sampling method

    An implementation of the deterministic gradient sampling method for the
    solution of nonsmooth, nonconvex optimization problems. It combines the
    method proposed in [GP2021,G2022] (for single-objective problems) with the
    bisection method from [G2024b].

    For arguments and return values, see
    https://github.com/b-gebken/DGS/blob/main/eps_descent_method.m

    [GP2021] Gebken, Peitz (2021): An Efficient Descent Method for Locally
    Lipschitz Multiobjective Optimization Problems. doi:
    0.1007/s10957-020-01803-w 
    [G2022] Gebken (2022): Computation and analysis of pareto critical sets in
    smooth and nonsmooth multiobjective optimization. doi: 10.17619/UNIPB/1-1327 
    [G2024b] Gebken (2024): A note on the convergence of deterministic gradient
    sampling in nonsmooth optimization. doi: 10.1007/s10589-024-00552-0
    """

    # Read input
    n = problem_data.x0.shape[0]
    f = problem_data.oracle[0]
    subgrad_f = problem_data.oracle[1]
    x0 = problem_data.x0

    eps_arr = algo_options['eps_arr']
    delta_arr = algo_options['delta_arr']
    rand_sample_N = algo_options['rand_sample_N']
    memory_max_size = algo_options['memory_max_size']
    c = algo_options['c']
    ls_flag = algo_options['ls_flag']
    max_iter = algo_options['max_iter']
    disp_flag = algo_options['disp_flag']

    # Initialization
    j_max = len(eps_arr)
    x_arr_list = []
    fx_arr_list = []
    num_sample_arr_list = []

    memory = Memory(memory_max_size)

    eval_counter = np.zeros(2)

    if disp_flag >= 1:
        start_time = time.time()
        print('Deterministic gradient sampling...')

    # Loop over all (eps,delta) pairs
    for j in range(j_max):

        if disp_flag >= 2:
            print('    Iteration j = ',j,'/',j_max,'...')
            print('        eps_j   = ',eps_arr[j])
            print('        delta_j = ',delta_arr[j])
            print('        Running descent iterations...')

        # Starting point is either x0 (if j = 1) or the final iterate of the previous descent sequence.
        if j == 0:
            x_arr = [x0]
            f_xi = f(x0); eval_counter[0] += 1
        else:
            x_arr = [x_arr_list[j-1][-1]]
            f_xi = f_x_new

        fx_arr = [f_xi]
        num_sample_arr = [0]

        if disp_flag >= 2:
            desc_start_time = time.time()
        
        # Descent loop for fixed eps and delta
        for i in range(max_iter):

            if (disp_flag >= 3) and (np.mod(i+1,100) == 0):
                tmp_counter = eval_counter[1]
                print('            Iteration i = ',i,' (j = ',j,')...')
                print('                Computing descent direction...')

            # Step 2 in [G2024a]
            v, f_eps_v, num_sample = descent_direction(x_arr[i],f_xi,f,subgrad_f,eps_arr[j],delta_arr[j],c,rand_sample_N,memory,eval_counter,disp_flag)

            if (disp_flag >= 3) and (np.mod(i+1,100) == 0):
                print('                    ...done!')
                print('                Req. subgrad. eval.: ',eval_counter[1] - tmp_counter)

            # Step 3 in [G2024a]
            if norm(v,ord=2) <= delta_arr[j]:
                # Step 4 in [G2024a]
                f_x_new = f_xi
                break
            # Step 5 in [G2024a]
            else:
                # Step 6 in [G2024a]
                if ls_flag == 'eps':
                    t = eps_arr[j]/norm(v,ord=2)
                    f_x_new = f_eps_v
                elif (ls_flag == 'armijo') or (ls_flag == 'armijo_normal'):
                    if ls_flag == 'armijo':
                        t = 1
                    elif ls_flag == 'armijo_normal':
                        t = 1/norm(v,ord=2)

                    if t <= eps_arr[j]/norm(v,ord=2):
                        t = eps_arr[j]/norm(v,ord=2)
                        f_x_new = f_eps_v
                    else:
                        f_x_new = f(x_arr[i] + t*v); eval_counter[0] += 1
                        while f_x_new - f_xi > -c*t*norm(v,ord=2)**2:
                            t = t/2

                            if t <= eps_arr[j]/norm(v,ord=2):
                                t = eps_arr[j]/norm(v,ord=2)
                                f_x_new = f_eps_v
                                break
                            
                            f_x_new = f(x_arr[i] + t*v); eval_counter[0] += 1
            
                # Step 7 in [G2024a]
                x_arr.append(x_arr[i] + t*v)
                fx_arr.append(f_x_new)
                num_sample_arr.append(num_sample)

                if (disp_flag >= 3) and (np.mod(i+1,100) == 0):
                    print('                norm(v,2)  = ',norm(v,ord=2),' (delta_j = ',delta_arr[j],', eps_j = ',eps_arr[j],')')
                    print('                New f val. = ',f_x_new)
                    print('                f decr.    = ',f_xi - f_x_new)

                f_xi = f_x_new
    
        if disp_flag >= 2:
            print('        ...done in N_j = ',len(x_arr)-1,' iterations (in ',time.time() - desc_start_time,'s).')

        x_arr_list.append(x_arr)
        fx_arr_list.append(fx_arr)
        num_sample_arr_list.append(num_sample_arr)

    x_opt = x_arr_list[-1][-1]
    f_opt = f_x_new
    
    if disp_flag >= 1:
        end_time = time.time()
        print('    ...done!')
        print('    Initial obj. value:  ',fx_arr_list[0][0])
        print('    Final obj. value:    ',f_x_new)
        print('    Total iterations:    ',sum([len(x_arr) for x_arr in x_arr_list]),' = ',sum([len(x_arr)-1 for x_arr in x_arr_list]),' + ',j_max)
        print('    Total f eval.:       ',eval_counter[0])
        print('    Total subgrad eval.: ',eval_counter[1])
        print('    Total time:          ',end_time - start_time)

    return x_opt, f_opt, x_arr_list, fx_arr_list, eval_counter, num_sample_arr_list
