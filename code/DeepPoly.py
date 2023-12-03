import torch
import torch.nn as nn


class FCVerifier:
    
    def __init__(self, weights_l: list, weights_u: list, biases_l: list, biases_u: list, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb # lower bound
        self.ub = ub # upper bound
        self.weights_l = weights_l
        self.weights_u = weights_u
        self.biases_l = biases_l
        self.biases_u = biases_u

    def direct_forwardpass(self) -> 'DeepPoly':
        weight_l = self.weights_l[-1]
        weight_u = self.weights_u[-1]

        for i in range(len(self.weights_l) - 1):
            weight_ll = torch.where(weight_l > 0, weight_l, weight_u)
            weight_uu = torch.where(weight_u > 0, weight_u, weight_l)
            weight_l = weight_ll @ self.weights_l[-i-2]
            weight_u = weight_uu @ self.weights_u[-i-2]
        
        bias_l = self.biases_l[0]
        bias_u = self.biases_u[0]
        for i in range(len(self.biases_l)-1):
            bias_l = bias_l.repeat(self.weights_l[i+1].shape[0], 1)
            bias_u = bias_u.repeat(self.weights_u[i+1].shape[0], 1)
            bias_l = (self.weights_l[i+1] * bias_l).sum(dim=1) + self.biases_l[i+1]
            bias_u = (self.weights_u[i+1] * bias_u).sum(dim=1) + self.biases_u[i+1]
        
        lb = self.lb.repeat(weight_l.shape[0], 1)
        ub = self.ub.repeat(weight_u.shape[0], 1)
        assert lb.shape == weight_l.shape
        assert ub.shape == weight_u.shape

        mul_lb = torch.where(weight_l > 0, lb, ub)
        mul_ub = torch.where(weight_u > 0, ub, lb)

        lb = (mul_lb * weight_l).sum(dim=1)
        ub = (mul_ub * weight_u).sum(dim=1)
        assert lb.shape == bias_l.shape
        assert ub.shape == bias_u.shape

        lb += bias_l
        ub += bias_u

        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)
        
        return DeepPoly(lb, ub)

class ReluVerifier:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb # lower bound
        self.ub = ub # upper bound
        
    def get_relu_weights(self) -> 'DeepPoly':
        # Follows from the rules in the lecture.
        bias_l = torch.zeros_like(self.lb)
        bias_u = torch.zeros_like(self.ub)
        for i in range(self.lb.size(dim=1)):
            if self.lb[0][i] >= 0:
                self.lb[0][i] = self.lb[0][i]
                self.ub[0][i] = self.ub[0][i]
            elif self.ub[0][i] <= 0:
                self.lb[0][i] = 0
                self.ub[0][i] = 0
            else:
                # self.lb[0][i] = 0
                self.lb[0][i] = self.lb[0][i]
                weight_i = self.ub[0][i] / (self.ub[0][i] - self.lb[0][i])
                bias_u[0][i] = - weight_i * self.lb[0][i]
                self.ub[0][i] = weight_i * self.ub[0][i]
        
        weight_lower = torch.diag(self.lb.reshape(-1))
        weight_upper = torch.diag(self.ub.reshape(-1))
        
        assert bias_l.shape == bias_u.shape == self.lb.shape == self.ub.shape
        
        return weight_lower, weight_upper, bias_l, bias_u

class DeepPoly:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb # lower bound
        self.ub = ub # upper bound 
    
    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'DeepPoly':
        lb = x - eps
        lb.clamp_(min=0, max=1)

        ub = x + eps
        ub.clamp_(min=0, max=1)

        return DeepPoly(lb, ub)
    
    def propagate_flatten(self, flatten: nn.Flatten) -> 'DeepPoly':
        lb = flatten(self.lb)
        ub = flatten(self.ub)
        return DeepPoly(lb, ub)

    def propagate_linear(self, fc: nn.Linear) -> 'DeepPoly':
        assert self.lb.shape == self.ub.shape
        assert len(self.lb.shape) == 2
        assert self.lb.shape[0] == 1 and self.lb.shape[1] == fc.weight.shape[1]
        
        # We want to have a copy (repeat) of the lower/upper bounds for each neuron in the next layer
        lb = self.lb.repeat(fc.weight.shape[0], 1)
        ub = self.ub.repeat(fc.weight.shape[0], 1)
        assert lb.shape == ub.shape == fc.weight.shape

        # When computing the new lower/upper bounds, we no longer need to take into account the sign of the weights as we want to retain the true 
        # symbolic representation of the linear layer.

        # mul_lb = torch.where(fc.weight > 0, lb, ub)
        # mul_ub = torch.where(fc.weight > 0, ub, lb)

        lb = (lb * fc.weight).sum(dim=1)
        ub = (ub * fc.weight).sum(dim=1)
        assert lb.shape == ub.shape == fc.bias.shape

        if fc.bias is not None:
            lb += fc.bias
            ub += fc.bias

        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

        return DeepPoly(lb, ub)

    def propagate_relu(self, relu: nn.ReLU) -> 'DeepPoly':
        # Follows from the rules in the lecture.
        lb = relu(self.lb)
        ub = relu(self.ub)
        return DeepPoly(lb, ub)

    def check_postcondition(self, y) -> bool:
        target = y
        target_lb = self.lb[0][target].item()
        for i in range(self.ub.shape[-1]):
            if i != target and self.ub[0][i] >= target_lb:
                return False
        return True