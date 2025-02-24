import torch
import torch.nn as nn
from ray.rllib.models.torch.misc import SlimFC

# split value branch since one is sequential and other is slimfc
def get_value_branches(network):
    value_branch = None
    value_branch_separate = None

    # value_branch is the output layer (w/ activation)
    if hasattr(network, '_value_branch'):
        value_branch = network._value_branch
        print("value branch found")
    
    # value_branch_separate is the input and hidden_layers that are not shared with the actor
    if hasattr(network, '_value_branch_separate'):
        value_branch_separate = network._value_branch_separate
        print("separate value branch found")
    
    return value_branch, value_branch_separate

value_branch, value_branch_sep = get_value_branches(self.critic_fcnet)


layers = []
def collect_layers(branch):
    for m in branch.children():
        # this handles the input and the inner hidden layers
        if isinstance(m, SlimFC):
            for layer in m._model.children():
                if isinstance(layer, torch.nn.Linear):
                    layers.append(layer)
        # this will handle the output layer
        elif isinstance(m, nn.Sequential):
            for layer in m.children():
                if isinstance(layer, torch.nn.Linear):
                    layers.append(layer)
        else:
            print("Adding no layers -- not SlimFC or Sequantial")
    return layers


'''Usage within RLLIB model's value branches'''
# initial_layers = collect_layers(value_branch_sep)
# final_layer = collect_layers(value_branch)
# layers = initial_layers + final_layer
# unpack each module sublass (Linear layers) using *
# self.layers = nn.Sequential(*layers)
