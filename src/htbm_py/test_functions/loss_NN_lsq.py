# Generates an oracle for the loss function of a simple neural network.
import torch
from torch import nn
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import numpy as np
from htbm_py.optimization_problem import OptimizationProblem

import matplotlib.pyplot as plt

data_fn = lambda var : torch.sin(torch.pi * torch.exp(var)) - var
loss_fn = lambda a,b,x,reg_param : 1/a.shape[0] * sum((a - b)**2) + reg_param*sum(x.abs())

def loss_NN_lsq(model,reg_param,N_data):

    data_x = torch.linspace(-1,1,N_data,dtype=torch.float64)
    data_y = data_fn(data_x)

    def f(x):
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        return loss_fn_x(x_torch).detach().cpu().numpy()
    
    def grad_f(x):
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        return grad_loss_fn_x(x_torch).detach().cpu().numpy()
    
    def hess_f(x):
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        return hess_loss_fn_x(x_torch).detach().cpu().numpy()
        
    def loss_fn_x(x_torch):
        params = unpack(x_torch,model)
        outputs = functional_call(model, params, (data_x.reshape(N_data,1),)).reshape(N_data)

        return loss_fn(outputs,data_y,x_torch,reg_param)

    def grad_loss_fn_x(x_torch):
        loss = loss_fn_x(x_torch)
        grad = torch.autograd.grad(loss, x_torch)[0]
        return grad
    
    def hess_loss_fn_x(x_torch):
        hess = torch.autograd.functional.hessian(loss_fn_x, x_torch)
        return hess

    n = sum(p.numel() for p in model.parameters())
    x0 = np.ones(n)

    problem_data = OptimizationProblem(
        oracle=[f,grad_f,hess_f],
        x0=x0
    )

    return problem_data

def unpack(x_torch, model):
    params = {}
    idx = 0

    for name, p in model.named_parameters():
        numel = p.numel()
        params[name] = x_torch[idx:idx+numel].view_as(p)
        idx += numel

    return params

def loss_unreg(x,model,N_data):
    """Returns the unregularized loss value.
    """

    data_x = torch.linspace(-1,1,N_data,dtype=torch.float64)
    data_y = data_fn(data_x)
    
    x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    params = unpack(x_torch,model)
    outputs = functional_call(model, params, (data_x.reshape(N_data,1),)).reshape(N_data)

    return loss_fn(outputs,data_y,x_torch,0).detach().cpu().numpy()

def visualize(x,model,N_data,**kwargs):
    """Visualize the model and the data.
    """

    data_x = torch.linspace(-1,1,N_data,dtype=torch.float64)
    data_y = data_fn(data_x)

    x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=False)

    N_plot = 1000
    plot_x = torch.linspace(-1.1,1.1,N_plot,dtype=torch.float64)

    params = unpack(x_torch,model)
    outputs_plot = functional_call(model, params, (plot_x.reshape(N_plot,1),)).reshape(N_plot)

    lw = 1.5
    ms = 10

    if 'fmt' in kwargs.keys():
        fmt = kwargs.get('fmt')
    else:
        fmt = 'r.--'

    if 'axes' in kwargs.keys():
        kwargs.get('axes').plot(data_x,data_y,fmt,markersize=ms,linewidth=lw)
        kwargs.get('axes').plot(plot_x,outputs_plot,'-',markersize=ms,linewidth=lw)
    else:
        plt.plot(data_x,data_y,fmt,markersize=ms,linewidth=lw)
        plt.plot(plot_x,outputs_plot,'-',markersize=ms,linewidth=lw)
        plt.show()

