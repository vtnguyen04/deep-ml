import torch
import torch.nn as nn


def capture_activations(model, layer_names, x):
    # TODO: register a forward hook on each named submodule that stores its
    #       detached output under that name
    # TODO: run the model on x
    # TODO: remove every hook, then return the dict of captured activations
    

    activations = {}    
    handles = []

    def make_hook(name: str):

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()

        return hook
    
    try: 
        for name in layer_names:

            submodule = model.get_submodule(name)
            hook_fn = make_hook(name)
            handle = submodule.register_forward_hook(hook_fn)
            handles.append(handle)
        _ = model(x)

    finally:
        for handle in handles:
            handle.remove()

    return activations