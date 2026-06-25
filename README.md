# Higher-order trust-region bundle method

A simple Python implementation of a trust-region bundle method using higher-order cutting-plane models (HTBM) for the solution of nonsmooth, nonconvex optimization problems. The trust-region subproblems are solved via _IPOPT_ (https://github.com/coin-or/Ipopt). The original Matlab implementation can be found at https://github.com/b-gebken/higher-order-trust-region-bundle-method, and the corresponding research articles are 

[GU2026a] Gebken, Ulbrich (2026): Superlinear convergence in nonsmooth optimization via higher-order cutting-plane models (https://arxiv.org/abs/2603.23236)<br/>
[GU2026b] Gebken, Ulbrich (2026): Enclosing minima in nonsmooth optimization via trust regions of higher-order cutting-plane models (https://arxiv.org/abs/2603.23261)<br/>

For testing purposes, Python implementations of BFGS and DGS (https://github.com/b-gebken/DGS) are included aswell.

## Experiments on training neural networks

In the following experiments, we consider optimization problems that arise in the training of neural networks (NNs) for regression, where the parameters of the network are optimized so that it maps given input data as close as possible to given output data. There are multiple potential sources of nonsmoothness in this case:
1. The model corresponding to the NN may be nonsmooth if nonsmooth activiation functions (like the ReLU function) are used in the NN.
2. The loss function for measuring the difference between the image of the model and the data may be nonsmooth.
3. A nonsmooth regularizer (like the L1-norm of the weights) may be added to the loss function.

There are two question we want to investigate with our experiments:
1. Does HTBM achieve superlinear convergence in this case?
2. What kind of (local) nonsmooth structure does the resulting (regularized) loss function have around its minima?

We emphasize that our first question is only of theoretical interest, as the HTBM (in its current form) is not well suited for this class of problems:
- Global convergence is typically more important than (fast) local convergence.
- The bundling procedure requires potentially many gradients and (full) Hessians to be stored.
- The trust-region (TR) subproblems that arise are expensive to solve.

As such, we only consider small NNs. Furthermore, for simplicity, we do not combine the HTBM with PyTorch by implementing the former as an Optimizer subclass. Instead, after creating the NN, we use the module src/htbm_py/test_functions/loss_NN.py to create an oracle that we can pass to a stand-alone version of the HTBM. This is likely far less efficient, but allows us to work with the standard version of the HTBM. 

The details of our experiments are as follows:
- We consider a feed-forward NN with 1-3-2-1 neurons (i.e., one neuron in the input layer, three neurons in the first hidden layer, two neurons in the second hidden layer, and one neuron in the output layer). The resulting number of weights (including biases) is $17$, which are denoted by as a vector $x \in \mathbb{R}^{17}$. For each set of weights $x$, we denote the resulting model function by $F(x,\cdot)$. For the activation functions we choose the ReLU function.
- For input data, we use equidistant points $t_i \in [-1,1]$, $i \in \\{1,\dots,N_\{data\}\\}$, and for the output data, we use $y_i = \sin(\pi \exp(t_i)) - t_i$. (See the images below for a visualizaton.)
- For the HTBM, we use the same parameters as in [GU2026a], Section 6, with $\varepsilon_{thr} = 10^{-6}$ as the smallest TR radius. Since we are only concerned with local convergence and the local nonsmooth structure, we choose an initial TR radius of $10^{-2}$ and an initial point $x^1$ whose distance to an approximated minimum (x_min in the code) is at most $10^{-3}$. (The approximated minima were computed beforehand through combination of DGS, BFGS and HTBM.) The best point found by the run of the HTBM is denoted by $x^{{best}}$. (For analyzing the speed of convergence, we will plot the values $\log_{10}(\\| x^j - x^{{best}} \\|_2)$. Since this is not well defined when $x^j = x^{{best}}$, a vertical dashed line is plotted for such $j$.) To make the behavior of the HTBM as easy to interpret as possible, we do not use random initial points for the bundle and do not reuse bundle information from memory (cf. [GU2026a], Remark 4.2(a)).
- For regularizing the loss function, we use the L1-norm $\\| x \\|\_1$ of the weights, multiplied by a regularization parameter $\lambda \geq 0$.
- The number of data points $N_{data}$, the type of loss function, and the regularization parameter $\lambda$ vary throughout the experiments.

### Experiments with the mean squared error loss

We first consider the case where the loss function is the _mean squared error_ (MSE), such that the overall objective functions is $f(x) = \frac{1}{N_{{data}}} \sum_{i = 1}^{N_{{data}}} \\| F(x,t_i) - y_i \\|\_2^2 + \lambda \\| x \\|_1$. In the first experiment, we consider $N\_{data} = 7$ data points and $\lambda = 10^{-4}$. The results are shown in Fig. 1.

<p>
  <img src="experiments/experiment_1/plot_1.png"/>
  <br/>
  <strong>Figure 1.</strong>
</p>

Before interpreting the results, we first discuss our visualization:
- The top-left plot shows the convergence behavior of $(x^j)_j$ w.r.t. to the index $j$. By the theory of [GU2026a], under certain assumptions, this sequence converges locally R-superlinearly to a minimum.
- The top-middle plot shows the convergence behavior of the objective values w.r.t. the number of oracle calls, where $j(l)$ denotes the index $j$ of the HTBM in which the $l$-th oracle call occured.
- The top-right plot shows the (connected) data points $(t_i,y_i)$ in red and the graph of the best-fit model $F(x^{{best}},\cdot)$ in blue.
- The bottom-left plot shows the size of the bundle in each iteration of the HTBM that was required to build a "good" higher-order cutting-plane model. (If this size is 1, then HTBM simply performs iterations of the classic trust-region Newton method.)
- The bottom-middle plot shows the relation of the distance of the solution of the TR subproblem to the TR radius (i.e., a value less than 1 means that the TR radius is inactive in this iteration.)
- Finally, the bottom-right plot shows, for each iterate $x^j$, the fraction $\frac{f(x^j) - f(x^{{best}})}{\\| x^j - x^{{best}} \\|\_2^2}$ plotted against $\log_{10}(\\| x - x^{{best}} \\|\_2)$, which can be used to get a rough idea of the order of growth of the objective function. However, since the solution may not be unique or even isolated, this plot has limited meaning. 

Considering the specific results in Fig. 1, the top-left plot suggests (roughly) R-superlinear convergence of $(x^j)_j$, with significant decrease starting at $x^{9}$. As expected from the theory, the significant decrease starts exactly when the TR becomes inactive and the bundle size increases.

The fact that the bundle size is larger than one also suggests that $f$ is nonsmooth around the minimum. This is not surprising, since $\lambda > 0$ means that $f$ inherits the nonsmoothness of the L1-norm. To be able to detect the possible nonsmoothness of $f$ that is caused be the nonsmoothness of the activation function, we next consider the case where $\lambda = 0$. The results are shown in Fig. 2. 

<p>
  <img src="experiments/experiment_1/plot_2.png"/>
  <br/>
  <strong>Figure 2.</strong>
</p>



<h1>Acknowledgements</h1>
This research was funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Projektnummer 545166481.
