import numpy as np
import time

def linesearch_armijo_wolfe(x,p,f_x,grad_f_x,eval_counter,problem_data,algo_options):
    """Inexact Wolfe line search

    The inexact Wolfe line search from [Lewis, Overton (2013)], Alg. 4.6,
    without differentiability check. A stopping criterion is added for numerical
    reasons, which stops when the step length becomes too small.
    """

    f = problem_data.oracle[0]
    grad_f = problem_data.oracle[1]

    c1 = algo_options['c1']
    c2 = algo_options['c2']
    step_threshold = algo_options['step_threshold']

    s = np.dot(grad_f_x,p)

    A = lambda t, f_xtp : f_xtp - f_x < c1*s*t
    W = lambda grad_f_xtp : np.dot(grad_f_xtp,p) > c2*s

    alpha = 0
    beta = np.inf
    t = 1

    while True:
        f_xtp = f(x + t*p)
        eval_counter[0] += 1
        if ~A(t,f_xtp):
            beta = t
        else:
            grad_f_xtp = grad_f(x + t*p)
            eval_counter[1] += 1
            if ~W(grad_f_xtp):
                alpha = t
            else:
                break

        if ~np.isinf(beta):
            t = (alpha+beta)/2
        else:
            t = 2*alpha
        
        if t <= step_threshold:
            t = 0
            f_xtp = f_x
            grad_f_xtp = grad_f_x
            break

    return t, f_xtp, grad_f_xtp

def bfgs(problem_data,algo_options):
    """BFGS method

    A simple implementation of the BFGS method (see, e.g., Alg. 6.1 in [Nocedal,
    Wright (2006)]). For the BFGS update, the formula (2.2) from [Lewis, Overton
    (2013)] is used. The stopping criteria are the step length being too small
    or the descent being too shallow.
    """

    # Read inputs
    f = problem_data.oracle[0]
    grad_f = problem_data.oracle[1]
    x0 = problem_data.x0
    n = x0.shape[0]
    
    N_iter = algo_options['N_iter']
    H0 = algo_options['H0']
    reset_period  = algo_options['reset_period']
    step_threshold  = algo_options['step_threshold']
    descent_threshold  = algo_options['descent_threshold']
    disp_flag  = algo_options['disp_flag']

    # Initialization
    eval_counter = np.zeros(2)
    
    x_arr = [x0]
    f_arr = [f(x0)]
    eval_counter[0] += 1
    grad_arr = [grad_f(x0)]
    eval_counter[1] += 1

    t_arr = []
    y_arr = []
    s_arr = []
    p_arr = []

    H_arr = [H0]

    if disp_flag > 0:
        print('Running BFGS method...')
        start_time = time.time()
        print_iter_arr = np.arange(0,N_iter,200)
    
    # Loop over k
    for k in range(N_iter):

        if (disp_flag > 1) and (k in print_iter_arr):
            print('    ',k,'/',N_iter,', f(x) = ',f_arr[k])

        # Compute search direction
        p_arr.append(-H_arr[k] @ grad_arr[k])

        # Compute Wolfe step length
        t, f_tmp, grad_tmp = linesearch_armijo_wolfe(x_arr[k],p_arr[k],f_arr[k],grad_arr[k],eval_counter,problem_data,algo_options)
        t_arr.append(t)
        f_arr.append(f_tmp)
        grad_arr.append(grad_tmp)

        # Update iterate
        x_arr.append(x_arr[k] + t_arr[k]*p_arr[k])

        # Update H via inverse BFGS update
        y_arr.append(grad_arr[k+1] - grad_arr[k])
        s_arr.append(x_arr[k+1] - x_arr[k])

        if t > 0:
            V = np.eye(n) - 1/np.dot(p_arr[k],y_arr[k]) * np.outer(p_arr[k],y_arr[k])
            H_arr.append(
                V @ H_arr[k] @ np.transpose(V) + t_arr[k] * 1/np.dot(p_arr[k],y_arr[k]) * np.dot(p_arr[k],p_arr[k])
            )
        else:
            H_arr.append(np.nan * np.ones([n,n]))


        # Ensure symmetry
        H_arr[k+1] = (H_arr[k+1] + np.transpose(H_arr[k+1]))/2

        # Restarts, if reset_period < Inf.
        if np.mod(k,reset_period) == 0:
            H_arr[k+1] = H0

        # Stopping criterion (step length)
        if t_arr[k] < step_threshold:
            if disp_flag > 0:
                print('    Stopped because step length too small. (t_k = ',t_arr[k],')')

            break
            
        # Stopping criterion (descent)
        if -np.dot(grad_arr[k],H_arr[k] @ grad_arr[k]) > -descent_threshold:
            if disp_flag > 0:
                print('    Stopped because descent too shallow. (∇f(x^k)^T * p^k = ',np.dot(grad_arr[k],p_arr[k]),')')

            break

    # Prepare output
    result_bfgs = {
        'x_arr': x_arr,
        'f_arr': f_arr,
        'grad_arr': grad_arr,
        'H_arr': H_arr,
        't_arr': t_arr,
        'y_arr': y_arr,
        's_arr': s_arr,
        'p_arr': p_arr,
        'eval_counter': eval_counter
    }

    if disp_flag > 0:
        end_time = time.time()
        print('    f value   = ',f_arr[k])
        print('    norm grad = ',np.linalg.norm(grad_arr[k]))
        print('    iters     = ',k)
        print('    time      = ',end_time - start_time)
        print('    f eval    = ',eval_counter[0])
        print('    grad eval = ',eval_counter[1])

    return result_bfgs