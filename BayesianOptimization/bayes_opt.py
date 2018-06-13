from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from gpytorch.module import Module
from gpytorch.likelihoods import GaussianLikelihood
from torch import optim
import torch

from .discrete_mes import DiscreteMES
from .discrete_ei import DiscreteEI
from .discrete_ucb import DiscreteUCB
from .discrete_pi import DiscretePI

class BayesianOptimization(Module):
    def __init__(self, GPModel, 
                 likelihood, 
                 optimizer
                 target, 
                 search_space, 
                 dimensions, 
                 acq_func="discrete_mes", 
                 acq_func_kwargs=None):
        
        if not isinstance(GPModel, Module):
            raise RuntimeError("BayesianOptimization can only handle Module")
        #check trained model
        for param in GPModel.parameters():
            assert param.grad is not None 
            assert param.grad.norm().item()> 0
        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError("BayesianOptimization can only handle GaussianLikelihood")
        if not isinstance(optimizier, optim):
            raise RuntimeError("BayesianOptimization can only handle torch optimizer")
            
        allowed_acq_funcs = ["discrete_mes", "discrete_ei", "discrete_ucb", "discrete_pi"]
        if acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), acq_func))

        self.model = GPModel
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.acq_func = acq_func
        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.acq_func_kwargs = acq_func_kwargs
        #negative wrapper to maximize the target function
        self.function = lambda x: -1*target(x)
        
        self._x_samples = None
        self._y_samples = None
        super(BayesianOptimization, self).__init__()
        
    def _generate_x_samples(search_space):
        self._x_samples
        self._y_samples = self.function(self._x_samples)
        raise NotImplementedError
        
    def update_model(next_point):
        train_x = Variable(torch.cat((self.model.train_inputs, next_point),dim=0))
        train_targets = Variable(torch.cat((self.model.train_targets, self.function(next_point)),dim=0))
        train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in (train_x,))
        self.model.set_train_data(train_inputs, train_targets, strict=False)
        
        self.model.train()
        self.likelihood.train()
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 10
        for i in range(training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_targets)
            loss.backward()
            
            print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                i + 1, training_iter, loss.data[0],
                self.model.covar_module.log_lengthscale.data[0, 0],
                self.model.likelihood.log_noise.data[0]
            ))
            
            self.optimizer.step()

        
    def step(self):
        if self.acq_func == "discrete_mes":
            nK = acq_func_kwargs.get("nK", 100)
            self.acq_func = DiscreteMES(self.model, nK)
        elif self.acq_func == "discrete_ei":
            self.acq_func = DiscreteEI(self.model, self.model.train_targets)
        elif self.acq_func == "discrete_pi":
            self.acq_func = DiscretePI(self.model, self.model.train_targets)
        elif self.acq_func == "discrete_ucb":
            kappa = acq_func_kwargs.get("kappa", 5)
            self.acq_func = DiscreteUCB(self.model, kappa)
        #next point to query
        _,next_point = self.acq_func(self._x_samples)
        return next_point
         
    def optimal(n_calls):
        for _ in range(n_calls):
            next_point = self.step()
            self.update_model(next_point)
    
    @property 
    def x_star(self):
        return self._x_samples[torch.argmax(self._f_samples)].view(1)
        
    @property 
    def y_star(self):
        return torch.max(self._f_samples)
        
    @property 
    def x_samples(self):
        return self._x_samples
        
    @property 
    def y_samples(self):
        return self._y_samples
    
    
    