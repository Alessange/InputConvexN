import polyscope as ps
import igl
import numpy as np
import matplotlib.pyplot as plt

aspect_ratio = 8
ns = 8
X, F = igl.triangulated_grid(aspect_ratio * ns + 1, ns + 1)
X[:, 0] *= aspect_ratio
X /= ns


X_np = X.astype(np.float64)
F_np = F.astype(np.int32)

n_verts = X_np.shape[0]

# point right
basis_left = np.tile(np.array([1.0, 0.0]), (n_verts, 1))

# point down
basis_down = np.tile(np.array([0.0, -1.0]), (n_verts, 1))

# point coswise1
y_curve = (np.cos(np.pi * X[:,0])) / (2 * ns)
basis_cos1 = np.zeros((n_verts,2))
basis_cos1[:,1] = y_curve #np.where(X[:,1] >= y_curve, 1.0, -1.0)

# point coswise2
y_curve = (np.cos(np.pi * 8 * X[:,1])) / (2 * ns)
basis_cos2 = np.zeros((n_verts,2))
basis_cos2[:,0] = y_curve #np.where(X[:,1] >= y_curve, 1.0, -1.0)





ps.init()
ps.set_ground_plane_mode("shadow_only")
ps_mesh = ps.register_surface_mesh("cantilever beam", X_np, F_np)
# ps_mesh.add_vector_quantity("point right", basis_left, defined_on="vertices", enabled=False)
# ps_mesh.add_vector_quantity("point cos1", basis_cos1, defined_on="vertices", enabled=False)
# ps_mesh.add_vector_quantity("point down", basis_down, defined_on="vertices", enabled=False)
# ps_mesh.add_vector_quantity("point cos2", basis_cos2, defined_on="vertices", enabled=False)

ps_mesh = ps.register_surface_mesh("cantilever beam", X_np + basis_cos2, F_np)

ps.show()


