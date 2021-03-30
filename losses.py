import numpy as np
import torch

def llp_bce(output, target):
    # BCE loss takes log of output, clamp values to avoid log(0)
    small_clamp = 1e-5
    output = output.clamp(min=small_clamp,max=1-small_clamp)
    avg_out = torch.mean(output)
    avg_tar = torch.mean(target)

    # Normalization term is undefined at 0.5, so define bounds around
    # this for the analytic expression and taylor approximation
    taylor_bound = 1e-2
    lo = 0.5 - taylor_bound
    hi = 0.5 + taylor_bound

    # Calculate normalization term, using analytic expression away from
    # 0.5 and taylor approximation close to 0.5
    if torch.le(avg_out, lo) | torch.gt(avg_out, hi):
        arctanh = 0.5*torch.log(1.0 / avg_out - 1.0)
        log_C = torch.log(2.0*arctanh / (1.0 - 2.0*avg_out))
    else:
        log_C = np.log(2.0) + 4.0/3.0*torch.pow(avg_out-1.0/2.0,2) + \
            104.0/45.0*torch.pow(avg_out-1.0/2.0,4)
    
    # Calculate BCE loss from continuous bernoulli distribution
    L = -avg_tar*avg_out.log() - (1 - avg_tar) * (1 - avg_out).log() - log_C

    try:
        del arctanh, output, target, avg_out, avg_tar, log_C
    except:
        del output, target, avg_out, avg_tar, log_C

    return L

