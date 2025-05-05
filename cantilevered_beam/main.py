from mesh import CantileverMesh
from transform import ModalBasis
from integrator import ImplicitIntegrator
from visualizer import PolyscopeViewer
import torch
import polyscope as ps

def run():
    mesh   = CantileverMesh(aspect_ratio=4, ns=20)
    basis  = ModalBasis(mesh, num_keep=25)
    

    # initial q, x, xdot
    X = torch.tensor(mesh.X_np, dtype=torch.float64, requires_grad=False)
    q0 = torch.zeros(basis.U.shape[1], dtype=torch.float64)
    x = X.clone().detach().requires_grad_(True)
    xdot0 = torch.zeros_like(x)

    fixed_idx = torch.where(X[:,0]==0)[0]
    integrator = ImplicitIntegrator(basis, mesh, dt=1/66, fixed_idx=fixed_idx)

    viewer = PolyscopeViewer(mesh)
    state = (q0, x, xdot0, 0.0)

    def callback():
        nonlocal state
        state = integrator.step(*state)
        viewer.update(state[1])

    ps.set_user_callback(callback)
    ps.show()

if __name__=="__main__":
    run()
