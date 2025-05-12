import torch
import numpy as np
import polyscope as ps
from mesh import CantileverMesh
from autoencoder import Autoencoder
import sys

all_disp = torch.load("cantilevered_beam/displacements.pt")
T, N, dim = all_disp.shape
X_flat = all_disp.view(T, -1).float()


model = Autoencoder(input_dim=X_flat.shape[1], hidden_dim=30, latent_dim=30)
model.load_state_dict(torch.load("cantilevered_beam/autoencoder_pca_init.pth", map_location="cpu"))
model.eval()

mesh = CantileverMesh(aspect_ratio=4, ns=20)
X0 = mesh.X_np 
F  = mesh.F

indices = list(range(T))
state = {"step": 0}


ps.init()

idx0 = indices[state["step"]]
V_orig0 = X0 + all_disp[idx0].numpy()
with torch.no_grad():
    rec0_flat = model(X_flat[idx0:idx0+1]).cpu().numpy().reshape(N, dim)
V_rec0 = X0 + rec0_flat

mesh_orig = ps.register_surface_mesh("Original",      V_orig0, F)
mesh_rec  = ps.register_surface_mesh("Reconstructed", V_rec0, F)


def callback():
    try:
        state["step"] = (state["step"] + 1) % T
        idx = indices[state["step"]]
        disp = all_disp[idx].numpy()
        V_o = X0 + disp
        mesh_orig.update_vertex_positions(V_o)
        with torch.no_grad():
            rec_flat = model(X_flat[idx:idx+1]).cpu().numpy().reshape(N, dim)
        V_r = X0 + rec_flat
        mesh_rec.update_vertex_positions(V_r)
    except IndexError:
        sys.exit(0)

ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()