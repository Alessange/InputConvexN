import trimesh
import polyscope as ps

mesh = trimesh.load('/home/towa/InputConvexNN/examples/cute_dragon.obj', process=False)

# Extract vertices and faces
V = mesh.vertices
F = mesh.faces

# Initialize Polyscope
ps.init()

# Register the mesh
ps.register_surface_mesh("my mesh", V, F)

# Show the viewer
ps.show()
