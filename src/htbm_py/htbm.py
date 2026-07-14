import numpy as np
import time

from htbm_py.optimization_problem import OptimizationProblem
from htbm_py.memory import Memory

import htbm_py.solve_subproblem_IPOPT as solve_subproblem_IPOPT

def sample_hypersphere(n):
    """Returns a random (uniformly sampled) point from the n-dim. unit sphere
    """

    point = np.random.normal(size=n)
    point = np.random.rand(1)**(1/n) * point/np.linalg.norm(point)

    return point

def generate_W(x,eps,f_x,c,reusing_eps_tolerance,memory,eval_counter,problem_data,algo_options,sp_options):
    """An implementation of the bundling Alg. 4.1 from [GU2026a]
    
    Includes the modification from [GU2026b] (see function htbm()) of adding the parameter c. Models
    of order q > 2 are currently not supported.
    """

    # Read problem specification
    n = x.shape[0]
    oracle = problem_data.oracle

    # Read algorithm options
    q = algo_options['q']
    sigma = algo_options['sigma']
    disp_flag = algo_options['disp_flag']

    init_N_sample = sp_options['init_N_sample']
    norm_flag = sp_options['norm_flag']
    sp_solver = sp_options['sp_solver']
    sp_solver_optns = sp_options['sp_solver_optns']

    # For computing numsample in the end
    eval_counter_old = eval_counter.copy()

    # Tolerance for reusing points
    if eps < 2*reusing_eps_tolerance:
        print("Warning: eps in magnitude of reuse tolerance (eps = ",eps,", reuse tolerance = ",reusing_eps_tolerance,").")

    # Reuse memorized elements
    if len(memory.sample_pts) > 0:        
        inds = np.linalg.norm(np.stack(memory.sample_pts) - x,axis=1,ord=norm_flag) <= eps + reusing_eps_tolerance

        W = [v for v, inds in zip(memory.sample_pts,inds) if inds]
        oracle_W = [v for v, inds in zip(memory.oracle_vals,inds) if inds]

        N_reused = len(W)

        if disp_flag >= 2:
            print("                    Reusing ",N_reused," sample points. (Memory size = ",len(memory.sample_pts),".)")
    else:
        W = []
        oracle_W = []

        N_reused = 0

    # Sample new elements in the trust-region
    if disp_flag >= 2:
        print("                    Initially sampling ",init_N_sample," point(s).")

    W.append(x)
    oracle_W.append([f_x] + [h(x) for h in oracle[1:]])

    for k in range(1,init_N_sample):
        W.append(eps * sample_hypersphere(n) + x)
        oracle_W.append([h(W[N_reused + k]) for h in oracle])

    N_W = N_reused + init_N_sample

    for i in range(q+1):
        if i == 0:
            eval_counter[i] += init_N_sample-1
        else:
            eval_counter[i] += init_N_sample

    ################
    #### Step 1 ####
    ################
    while True:

        ################
        #### Step 2 ####
        ################

        match sp_solver:
            case 'IPOPT':
                z_bar, theta, mu = solve_subproblem_IPOPT.solve(W,[v[0] for v in oracle_W],[v[1] for v in oracle_W],[v[2] for v in oracle_W],x,eps,sp_solver_optns)

        # print(z_bar)

        if np.linalg.norm(z_bar - x,ord=norm_flag) > 1.5*eps:
            print("Warning: eps-constraint is heavily violated! eps = ",eps, ", ||z-x|| = ",np.linalg.norm(z_bar - x,ord=norm_flag),", frac = ",np.linalg.norm(z_bar - x,ord=norm_flag)/eps)

        f_z_bar = oracle[0](z_bar)
        eval_counter[0] += 1

        ################
        #### Step 3 ####
        ################

        if f_z_bar - theta <= np.min([c,eps**(q+sigma)]):

            ################
            #### Step 4 ####
            ################

            break
        else:
            
            ################
            #### Step 6 ####
            ################

            # Sample new information at z_bar
            N_W += 1
            W.append(z_bar)
            oracle_W.append([f_z_bar] + [h(z_bar) for h in oracle[1:]])
            for i in range(1,q+1):
                eval_counter[i] += 1

    if disp_flag >= 2:
        print("                ...Alg. 4.1 finished!")
        print("                ",N_W - init_N_sample - N_reused," additional sample points required.")
        print("                f(z_bar) = ",f_z_bar)
        print("                theta    = ",theta)
        print("                f(z_bar) - theta  = ",f_z_bar - theta)
        print("                eps^(q+sigma)     = ",eps**(q+sigma))

    # Store newly sampled elements
    if memory.max_size > 0:
        memory.add(W[N_reused:],oracle_W[N_reused:])

    numsample = eval_counter[1] - eval_counter_old[1]

    return z_bar, f_z_bar, mu, numsample

def local_method(x1,eps1,problem_data,algo_options):
    """Local higher-order trust-region bundle method

    An implementation of Alg. 4.2 from [GU2026a]. Models of order q > 2 are
    currently not supported. 

    For arguments and return values, see
    https://github.com/b-gebken/higher-order-trust-region-bundle-method/blob/main/Algorithms/local_method.m

    [GU2026a] Gebken, Ulbrich (2026): Superlinear convergence in nonsmooth
    optimization via higher-order cutting-plane models
    (https://arxiv.org/abs/2603.23236)
    """

    # Read problem specification
    n = problem_data.x0.shape[0]
    oracle = problem_data.oracle

    # Read algorithm options
    q = algo_options['q']
    p = algo_options['p']
    sigma = algo_options['sigma']
    disp_flag = algo_options['disp_flag']
    memory_max_size = algo_options['memory_max_size']

    local_options = algo_options['local_options']
    kappa = local_options['kappa']
    eps_thr = local_options['eps_thr']
    j_thr = local_options['j_thr']
    act_thr = local_options['act_thr']

    norm_flag = local_options['norm_flag']

    # For measuring runtime
    if disp_flag >= 1:
        start_time = time.time()

    # Initialize variables (j_max from solving eps_j = eps_thr for j)
    j_max = int(np.floor(1 + (np.log(np.log(eps_thr/eps1)/np.log(kappa) + 1))/np.log((q+sigma)/p)))
    x_arr = []
    f_arr = np.zeros(j_max+1)
    eps_arr = np.zeros(j_max)
    act_arr = np.zeros(j_max)
    mu_arr = np.zeros(j_max)
    numsample_arr = np.zeros(j_max)
    eval_counter = np.zeros(q+1)
    eval_counter_arr = [np.zeros(q+1)]
    success_flag = True
    
    # Initialize the memory
    memory = Memory(memory_max_size)

    x_arr.append(x1)
    f_x = oracle[0](x1)
    eval_counter[0] += 1
    f_x1 = f_x
    f_arr[0] = f_x

    ################
    #### Step 1 ####
    ################

    for j in range(j_max):

        ################
        #### Step 2 ####
        ################

        # j+1 due to Python indexing
        eps_arr[j] = eps1 * kappa**(((q+sigma)/p)**((j+1)-1) - 1)

        if disp_flag >= 2:
            print("            ----- Iteration j = ",j,"/",j_max," -------------------")
            print("                eps_j = ",eps_arr[j])

        ################
        #### Step 3 ####
        ################

        if disp_flag >= 2:
            print("                Applying Alg. 4.1...")

        # Execute Alg. 4.1 for generating the set W
        z_bar, f_z_bar, mu, numsample = generate_W(x_arr[j],eps_arr[j],f_x,np.inf,0,memory,eval_counter,problem_data,algo_options,local_options)

        # Save number of oracle calls up to this iteration
        eval_counter_arr.append(eval_counter.copy())

        # Save number of samples required
        numsample_arr[j] = numsample

        # Save multiplier of trust-region constraint
        mu_arr[j] = mu

        # Save activity of trust-region constraint
        act_arr[j] = np.linalg.norm(z_bar - x_arr[j],ord=norm_flag)/eps_arr[j]

        if disp_flag >= 2:
            print("                activity   = ",act_arr[j])
            print("                act_thresh = ",act_thr)
            print("                j     = ",j)
            print("                j_thr = ",j_thr)

        ################
        #### Step 4 ####
        ################

        # Update iterate
        x_arr.append(z_bar)
        f_arr[j+1]= f_z_bar
        f_x = f_z_bar

        # Check for nonsuccess (cf. Subsection 5.1)
        if (j >= j_thr) and (act_arr[j] > act_thr):
            f_arr = f_arr[:j]
            act_arr = act_arr[:j]
            mu_arr = mu_arr[:j]
            numsample_arr = numsample_arr[:j]
            success_flag = False

            break
    
    # Find point with smallest objective value. (Recall that Alg. 4.2 is not a descent method.)
    best_f_val = np.min(f_arr)
    I_best_f_val = np.argmin(f_arr)
    best_x = x_arr[I_best_f_val]

    if disp_flag >= 1:
        if success_flag:
            print("        Local method successful!")
        else:
            print("        Local method failed!")

        print("            f(x1)    = ",f_x1)
        print("            best_f   = ",best_f_val)
        print("            f(z_bar) = ",f_z_bar)

        print(" ")
        end_time = time.time()
        print("            Runtime = ",end_time - start_time,"s")
        print("            Required evaluations: ")
        print("                f    - ",eval_counter[0])
        print("                grad - ",eval_counter[1])
        if q > 1:
            print("                hess - ",eval_counter[2])
    
    result_local_method = {
        'x_arr': x_arr,
        'f_arr': f_arr,
        'eps_arr': eps_arr,
        'act_arr': act_arr,
        'mu_arr': mu_arr,
        'numsample_arr': numsample_arr,
        'best_x': best_x,
        'best_f_val': best_f_val,
        'eval_counter': eval_counter,
        'eval_counter_arr': eval_counter_arr,
        'success_flag': success_flag
    }

    return result_local_method

def global_method(problem_data,algo_options):
    """Global higher-order trust-region bundle method

    An implementation of Alg. 2 from [GU2026b]. Models of order q > 2 are
    currently not supported. 

    For arguments and return values, see
    https://github.com/b-gebken/higher-order-trust-region-bundle-method/blob/main/Algorithms/global_method.m

    [GU2026b] Gebken, Ulbrich (2026): Enclosing minima in nonsmooth optimization 
    via trust regions of higher-order cutting-plane models
    (https://arxiv.org/abs/2603.23261)
    """

    # Read problem specification
    x0 = problem_data.x0
    n = x0.shape[0]
    oracle = problem_data.oracle

    # Read algorithm options
    q = algo_options['q']
    p = algo_options['p']
    sigma = algo_options['sigma']
    disp_flag = algo_options['disp_flag']
    memory_max_size = algo_options['memory_max_size']
    local_flag = algo_options['local_flag']

    global_options = algo_options['global_options']
    delta_arr = global_options['delta_arr']
    tau_arr = global_options['tau_arr']
    c = global_options['c']
    norm_flag = global_options['norm_flag']
    i_max = global_options['i_max']

    # For measuring runtime
    if disp_flag >= 1:
        start_time = time.time()

    # Initialize variables
    j_max = len(delta_arr)
    x_cell = []
    numsample_cell = []
    numsample_arr = []
    eval_counter = np.zeros(q+1)
    result_local_phase = []

    # Initialize the memory
    memory = Memory(memory_max_size)

    x_arr = [x0]
    f_x = oracle[0](x0)
    eval_counter[0] += 1

    # Tolerance for determining whether memorized points lie in the current trust region.
    reusing_eps_tolerance = 1e-7

    ##################
    ##### Step 1 #####
    ##################

    # Outer j-loop
    for j in range(j_max):

        if disp_flag >= 2:
            print('----- Outer iteration j = ',j,' -------------------')
            print('    delta_j = ',delta_arr[j],', tau_j = ',tau_arr[j])

        ##################
        ##### Step 2 #####
        ##################

        # Inner i-loop
        for i in range(i_max):

            if disp_flag >= 2:
                print('    ----- Inner iteration j = ',j,', i = ',i,' -----')
                print('        delta_j = ',delta_arr[j],', tau_j = ',tau_arr[j])
                print('        Applying Alg. 4.1...')

            ##################
            ##### Step 3 #####
            ##################

            z_bar, f_z_bar, mu, numsample = generate_W(x_arr[i],delta_arr[j],f_x,c,reusing_eps_tolerance,memory,eval_counter,problem_data,algo_options,global_options)

            # Save number of sample points
            numsample_arr.append(numsample)

            if disp_flag >= 2:
                print('        f(x) = ',f_x)
                print('        f(x) - f(z) = ',f_x - f_z_bar)
                print('        (f(x) - f(z))/delta_j^p = ',(f_x - f_z_bar)/(delta_arr[j]**p))
                print('        tau_j                 = ',tau_arr[j])
                print('        ||z - x||/delta_j = ',np.linalg.norm(z_bar - x_arr[i],ord=norm_flag)/delta_arr[j])

            ##################
            ##### Step 4 #####
            ##################

            if (f_x - f_z_bar)/(delta_arr[j]**p) < tau_arr[j]:
                
                ##################
                ##### Step 5 #####
                ##################

                if disp_flag >= 2:
                    print('        Decision: Break i-loop.')

                break
            else:

                ##################
                ##### Step 7 #####
                ##################

                if disp_flag >= 2:
                    print('        Decision: Sufficient decrease.')

                x_arr.append(z_bar)
                f_x = f_z_bar

        # Save information of inner iteration
        x_cell.append(x_arr)
        numsample_cell.append(numsample_arr)

        ###################
        ##### Step 10 #####
        ###################

        if local_flag:
            if disp_flag >= 2:
                print('        Attempting local method...')
            
            result_local_method = local_method(x_arr[i],delta_arr[j],problem_data,algo_options)

            result_local_phase.append(result_local_method)
            eval_counter = [(eval_counter[i] + result_local_method['eval_counter'][i]) for i in range(q+1)]
            local_success_flag = result_local_method['success_flag']

            if local_success_flag:
                break

        else:
            if disp_flag >= 2:
                print('        Skipping local method.')
            
            local_success_flag = False
        
        ###################
        ##### Step 11 #####
        ###################

        x_arr = [x_cell[-1][-1]]
    
        if i == i_max:
            print('Warning: Maximal number of iterations reached before stopping criterion is satisfied')

    # Prepare output
    result_global_phase = {
        'local_success_flag ': local_success_flag,
        'x_cell': x_cell,
        'eval_counter': eval_counter,
        'numsample_cell': numsample_cell
    }
    
    if local_success_flag:
        result_global_phase['best_f_val'] = result_local_phase[-1]['best_f_val']
        result_global_phase['best_x'] = result_local_phase[-1]['best_x']
    else:
        result_global_phase['best_f_val'] = f_x
        result_global_phase['best_x'] = x_cell[-1][-1]

    if disp_flag >= 1:
        if local_success_flag:
            print('Algorithm finished in local phase')
            print('    Final objective value =',result_local_phase[-1]['best_f_val'])
        else:
            print('Algorithm finished outside of local phase')
            print('    Final objective value =',result_global_phase['best_f_val'])
        
        end_time = time.time()
        print('    Runtime               = ',end_time - start_time,'s')
        print('    Required evaluations: ')
        print('        f       - ',eval_counter[0])
        print('        subgrad - ',eval_counter[1])
        if q > 1:
            print('        subhess - ',eval_counter[2])

    return result_global_phase