import torch
import torch.nn as nn
from DeepPoly import DeepPoly

class AbstractBox:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb # lower bound
        self.ub = ub # upper bound 

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'AbstractBox':
        lb = x - eps
        lb.clamp_(min=0, max=1)

        ub = x + eps
        ub.clamp_(min=0, max=1)

        return AbstractBox(lb, ub)
    
    def propagate_flatten(self, flatten: nn.Flatten) -> 'AbstractBox':
        lb = flatten(self.lb)
        ub = flatten(self.ub)
        # return AbstractBox(lb, ub)
        return DeepPoly(lb, ub) # Return a DeepPoly object after the first layer to use the DeepPoly relaxation subsequently 

    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        assert self.lb.shape == self.ub.shape
        assert len(self.lb.shape) == 2
        assert self.lb.shape[0] == 1 and self.lb.shape[1] == fc.weight.shape[1]
        
        # We want to have a copy (repeat) of the lower/upper bounds for each neuron in the next layer
        lb = self.lb.repeat(fc.weight.shape[0], 1)
        ub = self.ub.repeat(fc.weight.shape[0], 1)
        assert lb.shape == ub.shape == fc.weight.shape

        # When computing the new lower/upper bounds, we need to take into account the sign of the
        # weight. Effectively, the expression that we want to overapproximate is:
        # x_1 * w_1 + x_2 * w_2 + ... + x_d * w_d,
        # where each x_i is overapproximated/abstracted by the box [lb_i, ub_i], i.e.
        # the concrete value of the neuron x_i can be any number from the interval [lb_i, ub_i].
        mul_lb = torch.where(fc.weight > 0, lb, ub)
        mul_ub = torch.where(fc.weight > 0, ub, lb)

        lb = (mul_lb * fc.weight).sum(dim=1)
        ub = (mul_ub * fc.weight).sum(dim=1)
        assert lb.shape == ub.shape == fc.bias.shape

        if fc.bias is not None:
            lb += fc.bias
            ub += fc.bias

        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

        return AbstractBox(lb, ub)

    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        # Follows from the rules in the lecture.
        lb = relu(self.lb)
        ub = relu(self.ub)
        return AbstractBox(lb, ub)

    def check_postcondition(self, y) -> bool:
        target = y
        target_lb = self.lb[0][target].item()
        for i in range(self.ub.shape[-1]):
            if i != target and self.ub[0][i] >= target_lb:
                return False
        return True