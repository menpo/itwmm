import numpy as np

from menpo.transform import Scale, UniformScale, Translation
from menpo3d.rasterize import rasterize_mesh

from itwmm.base import as_colouredtrimesh, instance_for_params


def l2_normalize(x, axis=0, epsilon=1e-12):
    """
    Transforms an `ndarray` to have a unit l2 norm along
    a given direction.
    ----------
    x : `ndarray`
        The array to be transformed.
    axis : `int`
        The axis that will be l2 unit normed.
    epsilon: `float`
        A small value such as to avoid division by zero.
    Returns
    -------
    x : (D,) `ndarray`
        The transformed array.
    """
    return x / np.maximum(np.linalg.norm(x, axis=axis), epsilon)


def mesh_in_unit_sphere(mesh):
    scale = UniformScale(1 / mesh.norm(), mesh.n_dims)
    translation = Translation(-scale.apply(mesh).centre())
    return translation.compose_after(scale)


def lambertian_shading(mesh, diffuse_colour=0.4,
                       albedo_weighting=0.5, ambient_colour=0.3,
                       light_positions=((2, -2, 1), (-2, 2, 1))):

    diffuse_colour = np.asarray(diffuse_colour)
    light_positions = l2_normalize(np.asarray(light_positions).reshape(-1, 3),
                                   axis=0)

    unit_transform = mesh_in_unit_sphere(mesh)
    mesh = unit_transform.apply(mesh)

    light_directions = l2_normalize(light_positions.reshape(-1, 1, 3) -
                                    mesh.points[None, ...], axis=0)

    # Calculate the lambertian reflectance for each light source.
    # This will be an `ndarray` of shape(num_light_sources, num_vertices)
    lambertian = np.sum(light_directions *
                        mesh.vertex_normals()[None, ...], 2)[..., None]

    # Sum up the contribution of all the light sources and multiply by the
    # diffusion colour.
    lambertian = lambertian.sum(0) * diffuse_colour + ambient_colour

    mesh.colours[...] = np.clip(mesh.colours * albedo_weighting +
                                lambertian * (1 - albedo_weighting),
                                0, 1)

    return unit_transform.pseudoinverse().apply(mesh)


def render_mesh_in_img(trimesh_3d_in_img, img_shape):
    [x_r, y_r, z_r] = trimesh_3d_in_img.range()
    av_xy_r = (x_r + y_r) / 2.0
    trimesh_3d_in_img = Scale([1, 1, av_xy_r / z_r]).apply(trimesh_3d_in_img)
    mesh_in_img_lit = lambertian_shading(as_colouredtrimesh(trimesh_3d_in_img))
    return rasterize_mesh(mesh_in_img_lit, img_shape)


def render_overlay_of_mesh_in_img(trimesh_3d_in_img, img):
    mesh_in_img = render_mesh_in_img(trimesh_3d_in_img, img.shape)
    return overlay_image(img, mesh_in_img.as_greyscale(), (0, 0))


def render_initialization(images, mm, id_indices, exp_indices, template_camera,
                          p, qs, cs, img_index):
    c_i = cs[img_index]
    q_i = qs[img_index]
    i_in_img = instance_for_params(mm, id_indices, exp_indices,
                                   template_camera,
                                   p, q_i, c_i)['instance_in_img']
    [x_r, y_r, z_r] = i_in_img.range()
    av_xy_r = (x_r + y_r) / 2.0
    i_in_img = Scale([1, 1, av_xy_r / z_r]).apply(i_in_img)
    mesh_in_img_lit = lambertian_shading(as_colouredtrimesh(i_in_img))
    return rasterize_mesh(mesh_in_img_lit, images[0].shape).as_unmasked()


def render_iteration(mm, id_ind, exp_ind, img_shape, camera, params,
                     img_index, iteration):
    params_i = params[iteration]
    c_i = params_i['cs'][img_index]
    p_i = params_i['p']
    q_i = params_i['qs'][img_index]
    i_in_img = instance_for_params(mm, id_ind, exp_ind, camera,
                                   p_i, q_i, c_i)['instance_in_img']
    [x_r, y_r, z_r] = i_in_img.range()
    av_xy_r = (x_r + y_r) / 2.0
    i_in_img = Scale([1, 1, av_xy_r / z_r]).apply(i_in_img)
    mesh_in_img_lit = lambertian_shading(as_colouredtrimesh(i_in_img))
    return rasterize_mesh(mesh_in_img_lit, img_shape).as_unmasked()


def overlay_image(background, overlay, offset_vector):
    output = background.copy()

    indices = overlay.indices() + offset_vector

    new_mask = np.zeros_like(background.pixels[0], dtype=np.bool)

    max_ = np.array(background.bounds()[1])
    in_img = ~np.logical_or(np.any(indices < 0, axis=1),
                            np.any(indices > max_, axis=1))
    new_mask[indices[in_img, 0], indices[in_img, 1]] = True
    output.pixels[:, new_mask] = overlay.as_vector(
        keep_channels=True)[:, in_img]
    return output
