import torch
from energy import stable_neohookean_elasticity, momentum_potential, bc_penalty

class ImplicitIntegrator:
    def __init__(self, basis, mesh, dt, fixed_idx, bc_weight=1.0):
        self.basis     = basis
        self.mesh      = mesh
        self.dt        = dt
        self.fixed     = fixed_idx
        self.bc_w      = bc_weight
        self.xdot      = None

    def step(self, q, x, xdot, t):
        X = torch.tensor(self.mesh.X_np, dtype=torch.float64)
        x = self.basis.to_full(q, X)
        xtilde = x + self.dt*xdot + self.dt**2 * torch.tensor([0,-9.8], dtype=torch.float64)

        def E(qr):
            xq = self.basis.to_full(qr, X)
            e_el = stable_neohookean_elasticity(xq, X, self.mesh.F, self.dt)
            e_mo = momentum_potential(xq, xtilde, self.mesh.mass)
            e_bc = bc_penalty(xq, X, self.fixed, self.bc_w)
            return 20*e_el + e_mo + e_bc * 11

        
        for _ in range(5):
            q = q.detach().clone().requires_grad_(True)
            E_val = E(q)
            g = torch.autograd.grad(E_val, q)[0]
            H = torch.autograd.functional.hessian(E, q)
            L, Q = torch.linalg.eigh(H)
            H_pd = Q @ torch.diag_embed(torch.abs(L)) @ Q.mH
            dq = torch.linalg.solve(H_pd, g)
            with torch.no_grad():
                q = (q - 1 *dq)
        x_new = self.basis.to_full(q, X)
        xdot_new = (x_new - x)/self.dt
        return q, x_new.detach(), xdot_new.detach(), t+self.dt
