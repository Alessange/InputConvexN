import igl
import scipy as sp
import numpy as np
import torch
import polyscope as ps
import torch.nn.functional as F
from skinning_eigenmode_basis import expand_skinning_eigenmodes

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
    global xdot, M, dt, fixed
    if x.shape != xtilde.shape:
        x = x.view(xtilde.shape)
    elastic_energy = stable_neohookean_elasticity(x, X, F, scale=dt**2)
    momentum_energy = momentum_potential(x, xtilde, M)
    
    boundary_condition = torch.sum(((x[fixed] - X[fixed])) ** 2)
    #print(boundary_condition, bc)
    
    total_energy = 20 * elastic_energy + momentum_energy + boundary_condition * 0.5
    
    #print(total_energy, boundary_condition * 10000000.)
    return total_energy

class Transformation:

    def __init__(self, U: torch.Tensor):

        self.U = U
        self.U_pinv = torch.linalg.pinv(U)

    def to_full(self, q: torch.Tensor) -> torch.Tensor:
        global X

        X_flat = self.U @ q 
        X_flat = X_flat.view(-1, 2) + X
        return X_flat

    def to_reduced(self, x: torch.Tensor) -> torch.Tensor:
    
        global X
        if x.ndim == 2:
            x_flat = x.reshape(-1)
        else:
            x_flat = x
        # project
        x_flat = x_flat - X.reshape_as(x_flat)
        q = self.U_pinv @ x_flat
        return q

def integrate():
    global t, q, xdot, g, fixed, bc, dt, trans, X

    print(f"Integrating t={t:.3f}", q.shape)

    # Compute the reduced coordinates
    x = trans.to_full(q) 
    xtilde = x + dt * xdot + dt**2 * g 
    #with torch.no_grad():
    #    xtilde[fixed] = bc + X[fixed]
        
    x_start = x.clone().detach()
    
    X = X.detach()
    
    #x = xtilde
    def energy_func_reduced_q(q_reduced):
            x_full = trans.to_full(q_reduced)
            return compute_total_energy(xtilde, x_full)
    
    for i in range(5):
        q = q.detach().clone().requires_grad_(True)
        print(q)
        #exit()
        

        E_val = energy_func_reduced_q(q)
        grad_q = torch.autograd.grad(E_val, q)[0]
        H_q    = torch.autograd.functional.hessian(energy_func_reduced_q, q)
        
        #print(H_q.shape)

        L, Q = torch.linalg.eigh(H_q)
        H_pd = Q @ torch.diag_embed(torch.abs(L)) @ Q.mH
        delta_q = torch.linalg.solve(H_pd, grad_q)

        with torch.no_grad():

            q = q - 1 * delta_q
            x_new = trans.to_full(q).view_as(x_start)
            #x_new[fixed] = bc + X[fixed]
            #q = trans.to_reduced(x_new)
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


if __name__ == "__main__":
    # Create a simple cantilevered beam
    aspect_ratio = 4
    ns =  20
    X,F = igl.triangulated_grid(aspect_ratio*ns+1,ns+1)
    M = igl.massmatrix(X,F,igl.MASSMATRIX_TYPE_VORONOI).diagonal()
    X[:,0] *= aspect_ratio
    X /= ns
    
    X_np = X.astype(np.float64)
    F_np = F.astype(np.int32)
    n_verts = X_np.shape[0]

    # Compute cotangent Laplacian (symmetric)
    L = igl.cotmatrix(X_np, F_np)

    # Compute the mass matrix (for generalized eigenproblem)
    M = igl.massmatrix(X_np, F_np, igl.MASSMATRIX_TYPE_VORONOI)


    num_eigen = 25

    # Solve the generalized eigenvalue problem: Lx = λMx
    vals, vecs = sp.sparse.linalg.eigsh(-L, k=num_eigen, M=M, sigma=0, which='LM')

    basis_expanded = expand_skinning_eigenmodes(vecs[:,:5], X_np)
    M = igl.massmatrix(X,F,igl.MASSMATRIX_TYPE_VORONOI).diagonal()
    M = torch.tensor(M,dtype=torch.float64).unsqueeze(1)
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


    # # Define basis fields
    # basis_left = np.tile([1.0, 0.0], (n_verts, 1))
    # basis_down = np.tile([0.0, -1.0], (n_verts, 1))
    # basis_cos1 = np.zeros((n_verts, 2))
    # basis_cos1[:,1] = (np.cos(np.pi * X_np[:,0])) / (2 * ns)
    # basis_cos2 = np.zeros((n_verts, 2))
    # basis_cos2[:,0] = (np.cos(np.pi * 8 * X_np[:,1])) / (2 * ns)

    # fields = [basis_left, basis_down, basis_cos1, basis_cos2]
    # U = np.stack([f.reshape(2*n_verts) for f in fields], axis=1)
    # U = torch.tensor(U, dtype=torch.float64)
    # X_flat = X_np.reshape(2*n_verts)

    # trans = Transformation(U) 
    # q = np.linalg.lstsq(U, X_flat, rcond=None)[0]
    # q = torch.tensor(q, dtype=torch.float64)
    # q = q.clone().detach().requires_grad_(True)


    U = torch.tensor(basis_expanded, dtype=torch.float64)
    trans = Transformation(U)

    q = trans.to_reduced(X) 
    #print(q.shape)
    q = torch.zeros_like(q)

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
        x_display = trans.to_full(q)
        #x_display[fixed] = bc + X[fixed] # boundary condition at displaying
        ps_mesh.update_vertex_positions(x_display.detach().numpy())

    ps.set_ground_plane_mode("shadow_only")
    ps.set_user_callback(callback)
    ps.show()
