import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autoencoder import Autoencoder

# Load displacement data and flatten
all_disp = torch.load("cantilevered_beam/displacements.pt")  # [T, N, dim]
T, N, dim = all_disp.shape
X_flat = all_disp.view(T, -1).float()                        # [T, N*dim]

# Load trained autoencoder
model = Autoencoder(input_dim=X_flat.shape[1], hidden_dim=500, latent_dim=5)
model.load_state_dict(torch.load("cantilevered_beam/autoencoder.pth", map_location="cpu"))
model.eval()

# Compute latent codes for each time-step
with torch.no_grad():
    Z = model.encoder(X_flat).cpu().numpy()  # shape: [T, latent_dim]

# 3D plot of latent trajectory using first three dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], marker='o')
ax.set_title("Latent Space Trajectory (Dims 0,1,2)")
ax.set_xlabel("Latent Dim 0")
ax.set_ylabel("Latent Dim 1")
ax.set_zlabel("Latent Dim 2")
plt.show()
