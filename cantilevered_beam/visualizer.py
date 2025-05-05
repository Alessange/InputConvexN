import polyscope as ps

class PolyscopeViewer:
    def __init__(self, mesh):
        ps.init()
        self.ps_mesh = ps.register_surface_mesh("cantilever", mesh.X_np, mesh.F)
        ps.set_ground_plane_mode("shadow_only")

    def update(self, x):
        self.ps_mesh.update_vertex_positions(x.numpy())
