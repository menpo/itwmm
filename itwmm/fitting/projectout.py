import numpy as np
from .base import sample_at_bc_vi, visible_sample_points


def project_out(J, U, fast_approx=True):
    tmp = J.dot(U.T if fast_approx else np.linalg.pinv(U))
    return J - tmp.dot(U)


def sample_uv_terms(instance, image, camera, mm, shape_pc, grad_x, grad_y,
                    n_samples):
    # subsample all the terms we need to compute a project out update.

    # Apply camera projection on current instance
    instance_in_image = camera.apply(instance)

    # Compute indices locations for sampling
    (vert_indices, bcoords, tri_indices,
     yx) = visible_sample_points(instance_in_image, image.shape, n_samples)

    # Warp the mesh with the view matrix (rotation + translation)
    instance_w = camera.view_transform.apply(instance.points)

    # Sample all the terms from the model part at the sample locations
    warped_uv = sample_at_bc_vi(instance_w, bcoords, vert_indices)

    # n_samples x n_channels x n_texture_comps
    U_tex_uv = mm.sample_texture_model(bcoords, tri_indices)

    # n_texture_comps x (n_samples * n_channels)
    U_tex_uv = U_tex_uv.reshape((-1, U_tex_uv.shape[-1])).T

    # n_channels x n_samples
    m_texture_uv = mm.instance().sample_texture_with_barycentric_coordinates(
        bcoords, tri_indices).T

    # n_samples x 3 x n_shape_components
    shape_pc_uv = (sample_at_bc_vi(shape_pc, bcoords, vert_indices)
                   .reshape([n_samples, 3, -1]))

    # Sample all the terms from the image part at the sample locations
    # img_uv: (channels, samples)
    img_uv = image.sample(yx)
    grad_x_uv = grad_x.sample(yx)
    grad_y_uv = grad_y.sample(yx)

    # Compute error
    # img_error_uv: (channels x samples,)
    img_error_uv = (m_texture_uv - img_uv).ravel()

    return (instance_w, instance_in_image, warped_uv, img_error_uv,
            shape_pc_uv, U_tex_uv, grad_x_uv, grad_y_uv)
