import polyscope as ps
import igl
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg


def expand_skinning_eigenmodes(basis, x):
    basis = basis.T  # transpose (1, 0) in torch -> .T in numpy
    basis = basis.reshape(basis.shape[0], basis.shape[1], 1)
    v00 = np.expand_dims(x[:, 0], axis=1)  # reshape(-1,1)
    v01 = np.expand_dims(x[:, 1], axis=1)
    
    

    v00 = np.tile(v00, (1, basis.shape[0])).T  # expand_as
    v01 = np.tile(v01, (1, basis.shape[0])).T
    v00 = v00.reshape(basis.shape)
    v01 = v01.reshape(basis.shape)
    

    cat_zeros = np.zeros_like(basis)
    #print(basis.shape)

    part_1 = np.concatenate((basis, cat_zeros), axis=-1)
    part_2 = np.concatenate((cat_zeros, basis), axis=-1)

    part_3 = np.concatenate((basis * v00, cat_zeros), axis=-1)
    part_4 = np.concatenate((cat_zeros, basis * v00), axis=-1)

    part_5 = np.concatenate((basis * v01, cat_zeros), axis=-1)
    part_6 = np.concatenate((cat_zeros, basis * v01), axis=-1)

    ret = np.concatenate((part_1, part_2, part_3, part_4, part_5, part_6), axis=0)
    ret = ret.reshape(ret.shape[0], -1)
    
    print(ret.shape)

    return ret.T  # transpose(1, 0)


if __name__ == '__main__':
    aspect_ratio = 8
    ns = 8
    X, F = igl.triangulated_grid(aspect_ratio * ns + 1, ns + 1)
    X[:, 0] *= aspect_ratio
    X /= ns


    X_np = X.astype(np.float64)
    F_np = F.astype(np.int32)

    n_verts = X_np.shape[0]



    # Compute cotangent Laplacian (symmetric)
    L = igl.cotmatrix(X_np, F_np)

    # Compute the mass matrix (for generalized eigenproblem)
    M = igl.massmatrix(X_np, F_np, igl.MASSMATRIX_TYPE_VORONOI)


    num_eigen = 10

    # Solve the generalized eigenvalue problem: Lx = λMx
    vals, vecs = scipy.sparse.linalg.eigsh(-L, k=num_eigen, M=M, sigma=0, which='LM')


    basis_expanded = expand_skinning_eigenmodes(vecs[:,:5], X_np)

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps_mesh = ps.register_surface_mesh("cantilever beam", X_np, F_np)

    # visualizing the scalar field of skinning eigenmodes in 'Fast Complementary Dynamics via Skinning Eigenmodes', not very important in this project
    #for idx in range(vecs.shape[-1]):
    #    ps_mesh.add_scalar_quantity("skinning mode " + str(idx), vecs[:,idx].reshape(-1))

    for idx in range(basis_expanded.shape[-1]):
        ps_mesh.add_vector_quantity("displacement " + str(idx), basis_expanded[:,idx].reshape(-1, 2))






    ps.show()


