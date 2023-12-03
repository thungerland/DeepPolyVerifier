import argparse
import torch
import torch.nn as nn

from networks import get_network
from utils.loading import parse_spec
from AbstractBox import AbstractBox
from DeepPoly import DeepPoly, FCVerifier as FCV, ReluVerifier as RV

DEVICE = "cpu"

def certify_sample(model, x, y, eps) -> bool:
    box = AbstractBox.construct_initial_box(x, eps)
    for layer in model:
        if isinstance(layer, nn.Linear):
            box = box.propagate_linear(layer)
        elif isinstance(layer, nn.Flatten):
            box = box.propagate_flatten(layer)
        elif isinstance(layer, nn.ReLU):
            box = box.propagate_relu(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box.check_postcondition(y)




def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    # TODO: Implement the verification procedure.
    box = DeepPoly.construct_initial_box(inputs, eps)
    weights_l = []
    weights_u = []
    biases_l = []
    biases_u = []

    for layer in net:
        if isinstance(layer, nn.Linear):
            # append linear weights and biases to the list 
            weights_l.append(layer.weight)
            weights_u.append(layer.weight)
            biases_l.append(layer.bias)
            biases_u.append(layer.bias)
        elif isinstance(layer, nn.Flatten):
            box = box.propagate_flatten(layer)
        elif isinstance(layer, nn.ReLU):
            # append Relu "weights" and "biases" to the list (if we look at the Relu bounds we can see they can be rephrased in terms of weights and biases)
            # We need to have the "box object" (or is this already fully backsubstituted?) up to this point to compute the Relu bounds
            fc_interim = FCV(weights_l, weights_u, biases_l, biases_u, box.lb, box.ub).direct_forwardpass()
            lb, ub = fc_interim.lb, fc_interim.ub
            RV_class = RV(lb, ub)
            relu_weight_l, relu_weight_u, relu_bias_l, relu_bias_u = RV_class.get_relu_weights()
            weights_l.append(relu_weight_l)
            weights_u.append(relu_weight_u)
            biases_l.append(relu_bias_l)
            biases_u.append(relu_bias_u)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        
    fcverifier = FCV(weights_l, weights_u, biases_l, biases_u, box.lb, box.ub)
    deep_poly = fcverifier.direct_forwardpass()
    
    # Should return True if the network is verified, False otherwise.
    # return certify_sample(net, inputs, true_label, eps)
    return deep_poly.check_postcondition(true_label)


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
