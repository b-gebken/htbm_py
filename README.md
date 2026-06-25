# Higher-order trust-region bundle method

A simple Python implementation of a trust-region bundle method using higher-order cutting-plane models (HTBM) for the solution of nonsmooth, nonconvex optimization problems. The trust-region subproblems are solved via _IPOPT_ (https://github.com/coin-or/Ipopt). The original Matlab implementation can be found at https://github.com/b-gebken/higher-order-trust-region-bundle-method, and the corresponding research articles are 

[GU2026a] Gebken, Ulbrich (2026): Superlinear convergence in nonsmooth optimization via higher-order cutting-plane models (https://arxiv.org/abs/2603.23236)<br/>
[GU2026b] Gebken, Ulbrich (2026): Enclosing minima in nonsmooth optimization via trust regions of higher-order cutting-plane models (https://arxiv.org/abs/2603.23261)<br/>

For testing purposes, Python implementations of BFGS and DGS (https://github.com/b-gebken/DGS) are included aswell.

## Experiments on training neural networks

In the following experiments, we consider optimization problems that arise in the training of neural networks (NNs) for regression, where the parameters of the network are optimized so that it maps given input data as close as possible to given output data. There are multiple potential sources of nonsmoothness in this case:
1. The model corresponding to the NN may be nonsmooth if nonsmooth activiation functions (like the ReLU function) are used in the NN.
2. The loss function for measuring the difference between the image of the model and the data may be nonsmooth.
3. A nonsmooth regularizer (like the L1-Norm of the weights) may be added to the loss function.

There are two question we want to investigate with our experiments:
1. Does HTBM achieve superlinear convergence in this case?
2. What kind of nonsmooth structure does the resulting (regularized) loss function have?

We emphasize that our first question is only of theoretical interest, as the HTBM (in its current form) is not well suited for this class of problems:
- Global convergence is typically more important than (fast) local convergence.
- The bundling procedure requires potentially many gradients and (full) Hessians to be stored.
- The trust-region (TR) subproblems that arise are expensive to solve.

As such, we only consider small NNs. Furthermore, for simplicity, we do not combine the HTBM with PyTorch by implementing the former as an Optimizer subclass. Instead, after creating the NN, we use the module src/htbm_py/test_functions/loss_NN.py to create an oracle that we can pass to a stand-alone version of the HTBM. This is likely far less efficient, but allows us to work with the standard version of the HTBM. 

The details of our experiments are as follows: We consider a feed-forward NN with 1-3-2-1 neurons (i.e., one neuron in the input layer, three neurons in the first hidden layer, two neurons in the second hidden layer, and one neuron in the output layer). The resulting number of weights (including biases) is $17$. For input data, we use equidistant points $t_i \in [-1,1]$, $i \in \\{1,\dots,N_\{data\}\\}$, and for the output data, we use $y_i = \sin(\pi \exp(t_i)) - t_i$. (See the images below for a visualizaton.) For the HTBM, we use the same parameters as in [GU2026a], Section 6, with $\varepsilon_{thr} = 10^{-6}$ as the smallest TR radius.

<h1>Acknowledgements</h1>
This research was funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Projektnummer 545166481.
