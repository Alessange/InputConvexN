import torch
from skinning_eigenmode_basis import expand_skinning_eigenmodes
from autoencoder import Autoencoder
class ModalBasis:
    def __init__(self, mesh, num_keep):
        vecs, _ = mesh.compute_eigenmodes(num_keep)
        U = expand_skinning_eigenmodes(vecs[:,:5], mesh.X_np)
        self.U       = torch.tensor(U, dtype=torch.float64)

    def to_full(self, q, X):
        disp = (self.U @ q).view(-1,2)
        return X + disp

class AutoencoderBasis:
    def __init__(self, input, hidden_dim=100, latent_dim=30, model=None):
        self.X = input
        self.latent_dim = latent_dim
        input_dim = input.flatten().shape[0]
        self.autoencoder = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.autoencoder.load_state_dict(torch.load(model, map_location="cpu"))
        self.autoencoder.eval()


    def to_full(self, q, X):
        x_recon = self.autoencoder.decoder(q).view(-1,2)
        return X + x_recon

