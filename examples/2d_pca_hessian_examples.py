import igl
import scipy as sp
import numpy as np
import torch
import polyscope as ps
import torch.nn.functional as F

# Create a simple cantilevered beam
aspect_ratio = 2
ns =  2
X,F = igl.triangulated_grid(aspect_ratio*ns+1,ns+1)
    # per-vertex mass
M = igl.massmatrix(X,F,igl.MASSMATRIX_TYPE_VORONOI).diagonal()
M = torch.tensor(M,dtype=torch.float64).unsqueeze(1)

X /= ns
X[:,0] *= aspect_ratio

X = torch.tensor(X,dtype=torch.float64,requires_grad=False)
F = torch.tensor(F,dtype=torch.int64)


x = X.clone().detach().requires_grad_(True)

fixed = torch.where(X[:, 0] == 0)[0]
bc = torch.zeros((fixed.shape[0], 2), dtype=torch.float64)


xdot = torch.zeros_like(x)
xprev = x.clone().detach()

dt = 1/66
t = 0
g = torch.tensor([0,-9.8],dtype=torch.float64)

lr = 0.005


def stable_neohookean_elasticity(x, X, F, scale):

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
    
    total_E = scale * torch.sum(A * W)
    return total_E
     
def momentum_potential(x, xtilde, mass):
    delta = x - xtilde
    energy = 0.5 * torch.sum(mass * (delta**2).sum(dim=1))
    
    return energy

def compute_total_energy(xtilde, x):
    global xdot, M, dt
    if x.shape != xtilde.shape:
        x = x.view(xtilde.shape)
    elastic_energy = stable_neohookean_elasticity(x, X, F, scale=dt**2)
    momentum_energy = momentum_potential(x, xtilde, M)
    total_energy = elastic_energy + momentum_energy 
    return total_energy

U = None
# Perform PCA
class Transformation:

    def __init__(self, U: torch.Tensor):

        self.U = U

    def to_full(self, q: torch.Tensor) -> torch.Tensor:

        return self.U @ q

    def to_reduced(self, x: torch.Tensor) -> torch.Tensor:

        return self.U.T @ x

trans = Transformation(U) 
q = trans.to_reduced(x)
q = q.clone().detach().requires_grad_(True)

def integrate():
    global t, q, xdot, g, fixed, bc, dt, trans, X

    print(f"Integrating t={t:.3f}")

    # Compute the reduced coordinates
    x = trans.to_full(q) 
    xtilde = x + dt * xdot + dt**2 * g 
    with torch.no_grad():
        xtilde[fixed] = bc + X[fixed]

    x_start = x.clone().detach()
    
    #x = xtilde
    def energy_func_reduced_q(q_reduced):
            x_full = trans.to_full(q_reduced)
            return compute_total_energy(xtilde, x_full)
    
    for i in range(50):
        q = q.detach().clone().requires_grad_(True)

        E_val = energy_func_reduced_q(q)
        grad_q = torch.autograd.grad(E_val, q)[0]
        H_q    = torch.autograd.functional.hessian(energy_func_reduced_q, q)

        L, Q = torch.linalg.eigh(H_q)
        H_pd = Q @ torch.diag_embed(torch.abs(L)) @ Q.mH
        delta_q = torch.linalg.solve(H_pd, grad_q)

        with torch.no_grad():

            q = q - 1.5 * delta_q
            x_new = trans.to_full(q).view_as(x_start)

        x = x_new.clone().detach().requires_grad_(True)
    damping = 1
    xdot = ((x - x_start) / dt * damping).detach()

    t += dt


# def show_initial_state():
#     ps.init()
#     ps.set_give_focus_on_show(True)
#     ps.register_surface_mesh("Initial State", X.detach().numpy(), F.detach().numpy())
#     ps.set_ground_plane_height_factor(0.125, is_relative=False)
#     ps.show()




ps.init()
ps.set_give_focus_on_show(True)
ps_mesh = ps.register_surface_mesh("x", x.detach().numpy(), F.detach().numpy())
#ps.set_ground_plane_height_factor(0.125, is_relative=False)

def callback():
    global t, x, xdot, xprev
    if False: #t > 5:
        t = 0
        x = trans.to_full(q).clone().detach().requires_grad_(True)
        xdot = torch.zeros_like(xdot)
        xprev = trans.to_full(q).view_as(xprev)
    integrate()
    x_full = trans.to_full(q)
    ps_mesh.update_vertex_positions(x_full.detach().numpy())

ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)

ps.show()




