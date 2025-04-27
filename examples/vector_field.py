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

# point coswise
y_curve = (np.cos(np.pi * X[:,0])) / (2 * ns)
basis_cos = np.zeros((n_verts,2))
basis_cos[:,1] = y_curve #np.where(X[:,1] >= y_curve, 1.0, -1.0)


y_curve = (1 + np.cos(np.pi * X[:,0])) / (2 * ns)
basis_old = np.zeros((n_verts,2))
basis_old[:,1] = np.where(X[:,1] >= y_curve, 1.0, -1.0)

ps.init()
ps.set_ground_plane_mode("shadow_only")
ps_mesh = ps.register_surface_mesh("cantilever beam", X_np, F_np)
ps_mesh.add_vector_quantity("point right", basis_left, defined_on="vertices", enabled=False)
ps_mesh.add_vector_quantity("point cos", basis_cos, defined_on="vertices", enabled=False)
ps_mesh.add_vector_quantity("point down", basis_down, defined_on="vertices", enabled=False)

# what does the basis look like?
ps_mesh = ps.register_surface_mesh("deformed beam", X_np + basis_cos, F_np)

ps_mesh = ps.register_surface_mesh("deformed beam_old", X_np + basis_old, F_np)

ps.show()


