import itertools
import torch
import torch.nn as nn


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def get_learnable_params(policy):
    if isinstance(policy, nn.Module):
        return (p for p in policy.parameters() if p.requires_grad)
    
    if isinstance(policy, dict):
        chains = []
        for m in policy.values():
            if isinstance(m, nn.Module):
                chains.append(p for p in m.parameters() if p.requires_grad)
        return itertools.chain.from_iterable(chains)
    
    if hasattr(policy, 'modules_dict') and isinstance(policy.modules_dict, dict):
        chains = []
        for m in policy.modules_dict.values():
            if isinstance(m, nn.Module):
                chains.append(p for p in m.parameters() if p.requires_grad)
        return itertools.chain.from_iterable(chains)
    
    return TypeError(f"Unsupported policy type: {type(policy)}")