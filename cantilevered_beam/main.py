from mesh import CantileverMesh
from transform import ModalBasis, AutoencoderBasis, InputConvexAutoencoderBasis
from integrator import ImplicitIntegrator
from visualizer import PolyscopeViewer
import torch
import polyscope as ps

def run():
    mesh   = CantileverMesh(aspect_ratio=4, ns=20)
    basis  = ModalBasis(mesh, num_keep=25)
    # basis  = AutoencoderBasis(mesh.X_np, hidden_dim=100, latent_dim=30, model="cantilevered_beam/autoencoder_pca_init.pth")
    basis = InputConvexAutoencoderBasis(mesh.X_np, hidden_dim=100, latent_dim=30, model="cantilevered_beam/autoencoder_pca_init_ic.pth")

    # initial q, x, xdot
    X = torch.tensor(mesh.X_np, dtype=torch.float64, requires_grad=False)
    # q0 = torch.zeros(basis.U.shape[1], dtype=torch.float64)
    q0 = torch.zeros(basis.latent_dim, dtype=torch.float32, requires_grad=True)
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
