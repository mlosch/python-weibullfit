try:
    import torch
    from weibull.backend_pytorch import fit
except ImportError:
    from weibull.backend_numpy import fit
