from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import itertools
from torch.autograd import Variable


class Space():
    def __init__(self):
        self.candidate_set = None

    @property
    def get_samples(self):
        return self.candidate_set


# upper_bound inclusive
class Real(Space):
    def __init__(self, lower, upper, sampling="linear", steps=1000):
        if upper <= lower:
            raise RuntimeError("the lower bound {} has to be less than the"
                               "upper bound {}".format(lower, upper))
        super(Real, self).__init__()
        if sampling == "linear":
            self.candidate_set = torch.linspace(lower, upper, steps)
        elif sampling == "log_uniform":
            self.candidate_set = torch.logspace(lower, upper, steps)
        else:
            raise ValueError("Sampling can only handle linear or log_uniform")


# upper_bound inclusive
class Integer(Space):
    def __init__(self, lower, upper):
        if upper <= lower:
            raise RuntimeError("the lower bound {} has to be less than the"
                               "upper bound {}".format(lower, upper))
        super(Integer, self).__init__()
        self.candidate_set = torch.range(lower, upper)


class Categorical(Space):
    def __init__(self, space):
        if len(space) < 1:
            raise RuntimeError("the number of class has to be greater than 0")
        super(Categorical, self).__init__()
        self.candidate_set = torch.Tensor(space)


class Dimension():
    def __init__(self, dimensions):
        if dimensions is None:
            raise RuntimeError("dimensions can't be none")
        self.candidate_set = []
        D = []
        for dimension in dimensions:
            D.append(dimension.get_samples)
        for e in itertools.product(*D):
            e = torch.stack(e)
            self.candidate_set.append(e)
        self.candidate_set = Variable(torch.stack(self.candidate_set)).cuda()

    @property
    def get_samples(self):
        return self.candidate_set
