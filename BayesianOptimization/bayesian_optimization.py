from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from gpytorch.module import Module
from gpytorch.likelihoods import GaussianLikelihood
from torch import optim
import torch
import gpytorch
from torch.autograd import Variable

from discrete_mes import DiscreteMES
from discrete_ei import DiscreteEI
from discrete_ucb import DiscreteUCB
from discrete_pi import DiscretePI
from dimension import Dimension
from torch.autograd import Variable

from matplotlib import pyplot as plt
from matplotlib import gridspec

# helper function: plotting gp for 1D
def plot_gp(train_x, train_y, x, y, rand_var=None, model=False, acq = None, bo=False):
    fig = plt.figure(figsize=(5,10))
    gs = gridspec.GridSpec(2,1,height_ratios=[2, 1]) 
    axis = plt.subplot(gs[0])
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(train_x, train_y, 'D', markersize=8, label=u'Observations', color='r')
    fig.suptitle('Gaussian Process and Acquisition Function',fontdict={'size':30})
    # if model is trained 
    if model == True:
        lower, upper = rand_var.confidence_region()
        mean = rand_var.mean().cpu().data.numpy()
        var = rand_var.var().cpu().data.numpy()
        axis.plot(x, mean, '--', color='k', label='Prediction')
       # axis.fill_between(x, lower.data.numpy(), upper.data.numpy(), alpha=.6, label='95% confidence interval')
        axis.fill_between(x, y - 1.96 * var , y + 1.96 * var, alpha=.3, label='95% confidence interval')
    
    # if acqusition function is available
    if bo == True:
        acqusition = plt.subplot(gs[1])
        acqusition.plot(x, acq.data.cpu().numpy(), label='Utility Function', color='purple')
        acqusition.plot(x[torch.argmax(acq)], torch.max(acq).data.cpu().numpy(), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        acqusition.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    
class BayesianOptimization(Module):
    def __init__(self, GPModel, 
                 likelihood, 
                 target, 
                 search_space, 
                 #dimensions, 
                 acq_func="discrete_mes", 
                 acq_func_kwargs=None):
        
        if not isinstance(GPModel, Module):
            raise RuntimeError("BayesianOptimization can only handle Module")

        #check trained model
        for param in GPModel.parameters():
            if param.grad is not None and param.grad.norm().item() == 0:
                raise RuntimeError("Model is not trained")
        
        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError("BayesianOptimization can only handle GaussianLikelihood")
   
        allowed_acq_funcs = ["discrete_mes", "discrete_ei", "discrete_ucb", "discrete_pi"]
        if acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), acq_func))
            
        super(BayesianOptimization, self).__init__()
        self.model = GPModel
        self.likelihood = likelihood
        self.acq_func = acq_func
        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.acq_func_kwargs = acq_func_kwargs
        #negative wrapper to maximize the target function
        self.function = lambda x: -1*target(x)
        
        self._x_samples = Dimension(search_space).get_samples.cuda()
        self._y_samples = Variable(self.function(self._x_samples)).cuda()


        
    def update_model(self, next_point):
        train_x = Variable(torch.cat((self.model.train_inputs[0], next_point))).cuda()
        train_targets = Variable(torch.cat((self.model.train_targets, self.function(next_point).view(-1)))).cuda()
        train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in (train_x,))
        self.model.set_train_data(train_inputs, train_targets, strict=False)
        
        self.model.train()
        self.likelihood.train()
        print("after: self.model.train_inputs",self.model.train_inputs[0])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

        training_iter = 10
        for i in range(training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_targets)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.data[0]))
  
            self.optimizer.step()

        
    def step(self,plot=False):
        if self.acq_func == "discrete_mes":
            nK = self.acq_func_kwargs.get("nK", 500)
            self.acq_func = DiscreteMES(self.model, nK)
        elif self.acq_func == "discrete_ei":
            self.acq_func = DiscreteEI(self.model)
        elif self.acq_func == "discrete_pi":
            self.acq_func = DiscretePI(self.model)
        elif self.acq_func == "discrete_ucb":
            kappa = self.acq_func_kwargs.get("kappa", 5)
            self.acq_func = DiscreteUCB(self.model, kappa)
        #next point to query
        self.model.eval()
        self.likelihood.eval()
        acq, next_point, observed_pred = self.acq_func(self._x_samples)
        if plot:
            plot_gp(self.model.train_inputs[0].view(-1).cpu().numpy(), 
                    self.model.train_targets.view(-1).cpu().numpy(), 
                    self._x_samples.view(-1).cpu().numpy(), 
                    self._y_samples.view(-1).cpu().numpy(), 
                    observed_pred, True, acq, True)

        return next_point
         
    def optimal(self, n_calls, plot=False):
        for _ in range(n_calls):
            next_point = self.step(plot)
            self.update_model(next_point)
    
    @property 
    def x_star(self):
        return self._x_samples[torch.argmax(self._y_samples)].view(-1)
        
    @property 
    def y_star(self):
        return -1*torch.max(self._y_samples)
        
    @property 
    def x_samples(self):
        return self._x_samples
        
    @property 
    def y_samples(self):
        return -1*self._y_samples
    
    
    