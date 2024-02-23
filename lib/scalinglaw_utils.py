import numpy as np

SEQ_LEN = 256
VAL_FREQ = 1000

def params(n_blocks, dim):
    return (12 * n_blocks * dim**2)# + (dim * 32768)

def flops_per_step(n_blocks, dim, batch_size, model_type, seq_len=SEQ_LEN):
    flops_per_token = 6 * (params(n_blocks, dim))
    if model_type == 'autoregressive':
        # Self-attn FLOPs are halved because of causal mask
        flops_per_token += (n_blocks * seq_len * dim)
    elif model_type == 'diffusion':
        flops_per_token += 2 * (n_blocks * seq_len * dim)
        # Extra forward pass on 25% of the data for self-conditioning
        flops_per_token *= (1. + (0.33 * 0.25))
    else:
        raise Exception()
    tokens_per_step = seq_len * batch_size
    return flops_per_token * tokens_per_step

def chinchilla_tokens_given_params(params):
    return 8e9 * (params / 400e6)

def chinchilla_params_given_flops(flops):
    return 400e6 * np.sqrt(flops / 1.92e19)

def chinchilla_tokens_given_flops(flops):
    return 8e9 * np.sqrt(flops / 1.92e19)

def power_law_fit(x, y):
    """y = a * x ^ b"""
    fit = np.polyfit(np.log(x), np.log(y), 1)
    def fit_fn(x_):
        return np.exp(np.poly1d(fit)(np.log(x_)))
    a = np.exp(fit[1])
    b = fit[0]
    return a, b, fit_fn

def power_law_plus_constant_fit(x, y):
    """y = y0 + a*x^b"""
    best_mse = float('inf')
    best_fit = None
    for y0 in np.linspace(0., np.min(y), 10_000):
        a, b, fit_fn = power_law_fit(x, y - y0)
        def new_fit_fn(x_, fit_fn_=fit_fn, y0_=y0):
            return fit_fn_(x_) + y0_
        mse = np.mean((y - new_fit_fn(x))**2)
        if mse < best_mse:
            best_mse = mse
            best_fit = (a, b, y0, new_fit_fn)
    return best_fit