"""
Miscellaneous PyTorch functions
"""

import torch
import torch.nn.functional as F

def cross_entropy(logits, targets):
    # Much faster than F.cross_entropy by avoiding the transpose
    logprobs = F.log_softmax(logits, dim=2)
    return -logprobs[
        torch.arange(logits.shape[0])[:,None],
        torch.arange(logits.shape[1])[None,:],
        targets
    ]

@torch.jit.script
def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    """KL(p||q)"""
    return (
        sigma_q.log() - sigma_p.log()
        + (sigma_p**2 + (mu_p - mu_q)**2)/(2*sigma_q**2)
        - 0.5
    )
