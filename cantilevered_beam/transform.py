import torch
from skinning_eigenmode_basis import expand_skinning_eigenmodes

class ModalBasis:
    def __init__(self, mesh, num_keep):
        vecs, _ = mesh.compute_eigenmodes(num_keep)
        U = expand_skinning_eigenmodes(vecs[:,:5], mesh.X_np)
        self.U       = torch.tensor(U, dtype=torch.float64)

    def to_full(self, q, X):
        disp = (self.U @ q).view(-1,2)
        return X + disp

