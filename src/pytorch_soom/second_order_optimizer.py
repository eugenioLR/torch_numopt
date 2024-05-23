import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.nn.utils.stateless import functional_call
import copy

class SecondOrderOptimizer(Optimizer):
    """
    """

    def step(self, x, y, closure=None):
        """
        """

    def update(self, model, x, y):
        """
        """