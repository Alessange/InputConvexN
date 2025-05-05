import torch

def stable_neohookean_elasticity(x, X, F, dt):

    X_tri = X[F]
    x_tri = x[F]
    M = torch.stack([X_tri[:, 1] - X_tri[:, 0], X_tri[:, 2] - X_tri[:, 0]], dim=2)
    m = torch.stack([x_tri[:, 1] - x_tri[:, 0], x_tri[:, 2] - x_tri[:, 0]], dim=2)
    
    M_inv = torch.inverse(M)
    
    F_tensor = torch.bmm(m, M_inv)
    det_M = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    A = 0.5 * torch.abs(det_M)
    Ic = torch.sum(F_tensor * F_tensor, dim=(1,2))
    detF = F_tensor[:, 0, 0] * F_tensor[:, 1, 1] - F_tensor[:, 0, 1] * F_tensor[:, 1, 0]

    youngs_modulus = 0.37e3
    poissons_ratio = 0.4
    mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    lam = youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    
    alpha = 1.0 + mu / lam
    
    W = mu / 2.0 * (Ic - 2.0) + lam / 2.0 * (detF - alpha)**2
    
    total_E = dt * dt * torch.sum(A * W)
    return total_E
     
def momentum_potential(x, xtilde, mass):
    delta = x - xtilde
    energy = 0.5 * torch.sum(mass * (delta**2).sum(dim=1))
    
    return energy

def bc_penalty(x, X, fixed_idx, weight=1.0):
    diff = x[fixed_idx] - X[fixed_idx]
    return 0.5*weight*torch.sum(diff*diff)