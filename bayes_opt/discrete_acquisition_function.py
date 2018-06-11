from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from gpytorch.module import Module
from .acquisition_function import AcquisitionFunction

class DiscreteAcquisitionFunction(AcquisitionFunction):
    def __init__(self,GPModel):
        super(DiscreteAcquisitionFunction, self).__init__()