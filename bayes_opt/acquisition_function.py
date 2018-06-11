from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from gpytorch.module import Module

class AcquisitionFunction(Module):
    
    def __init__(self, GPModel):
        super(AcquisitionFunction, self).__init__()
        self.GPModel = GPModel
        
    def forward(x):
        raise NotImplementedError