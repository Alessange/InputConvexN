import igl
import numpy as np
import scipy as sp
import torch

class CantileverMesh:
    def __init__(self, aspect_ratio, ns):
        X, F = igl.triangulated_grid(aspect_ratio*ns+1, ns+1)
        X[:,0] *= aspect_ratio
        X /= ns
        self.X_np = X.astype(np.float64)
        self.F    = F.astype(np.int32)

        # mass & laplacian
        Mmat = igl.massmatrix(self.X_np, self.F, igl.MASSMATRIX_TYPE_VORONOI)
        self.mass = Mmat.diagonal()
        self.mass = torch.tensor(self.mass, dtype=torch.float64)
        self.mass = self.mass.unsqueeze(1)
        self.L     = igl.cotmatrix(self.X_np, self.F)

    def compute_eigenmodes(self, num_modes):
        Mmat = igl.massmatrix(self.X_np, self.F, igl.MASSMATRIX_TYPE_VORONOI)
        vals, vecs = sp.sparse.linalg.eigsh(-self.L, k=num_modes, M=Mmat, sigma=0, which='LM')
        return vecs, vals
