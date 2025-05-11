import torch
import matplotlib.pyplot as plt
from autoencoder import Autoencoder


all_disp = torch.load("cantilevered_beam/displacements.pt")
T, N, dim = all_disp.shape
X_flat = all_disp.view(T, -1).float()


model = Autoencoder(input_dim=X_flat.shape[1], hidden_dim=500, latent_dim=30)
model.load_state_dict(torch.load("cantilevered_beam/autoencoder.pth", map_location="cpu"))
model.eval()


with torch.no_grad():
    Z = model.encoder(X_flat).cpu().numpy()


plt.figure()
plt.plot(Z[:, 0], Z[:, 1], marker='o')
plt.title("Latent Space Trajectory (Dimensions 0 vs 1)")
plt.xlabel("Latent Dim 0")
plt.ylabel("Latent Dim 1")
plt.show()