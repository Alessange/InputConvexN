import torch
import numpy as np
import polyscope as ps
from mesh import CantileverMesh
from autoencoder import Autoencoder


all_disp = torch.load("cantilevered_beam/displacements.pt")
T, N, dim = all_disp.shape
X_flat = all_disp.view(T, -1).float()


model = Autoencoder(input_dim=X_flat.shape[1], hidden_dim=500, latent_dim=30)
model.load_state_dict(torch.load("cantilevered_beam/autoencoder.pth", map_location="cpu"))
model.eval()

mesh = CantileverMesh(aspect_ratio=4, ns=20)
X0 = mesh.X_np 
F  = mesh.F

indices = torch.randperm(T)[:3].tolist()



ps.init()

for idx in indices:
    disp_orig = all_disp[idx].numpy()
    V_orig = X0 + disp_orig
    with torch.no_grad():
        rec_flat = model(X_flat[idx:idx+1]).cpu().numpy().reshape(N, dim)
    V_rec = X0 + rec_flat
    
    error = np.mean(np.abs(V_rec - V_orig))
    print(f"Mean absolute error for idx {idx}: {error:.6f}")

    ps.register_surface_mesh(f"Original_{idx}",      V_orig, F)
    ps.register_surface_mesh(f"Reconstructed_{idx}", V_rec,  F)

ps.show()
