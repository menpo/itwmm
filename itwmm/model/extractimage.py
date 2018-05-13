import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Scale
from menpo3d.rasterize import (
    rasterize_shape_image_from_barycentric_coordinate_images,
    rasterize_barycentric_coordinate_images
)

from itwmm.base import as_colouredtrimesh


def per_vertex_occlusion(mesh_in_img, err_proportion=0.0001, render_diag=600):

    [x_r, y_r, z_r] = mesh_in_img.range()
    av_xy_r = (x_r + y_r) / 2.0

    rescale = render_diag / np.sqrt((mesh_in_img.range()[:2] ** 2).sum())
    rescale_z = av_xy_r / z_r

    mesh = Scale([rescale, rescale, rescale * rescale_z]).apply(mesh_in_img)
    mesh.points[...] = mesh.points - mesh.points.min(axis=0)
    mesh.points[:, :2] = mesh.points[:, :2] + 2
    shape = np.round(mesh.points.max(axis=0)[:2] + 2)

    bc, ti = rasterize_barycentric_coordinate_images(mesh, shape)
    si = rasterize_shape_image_from_barycentric_coordinate_images(
        as_colouredtrimesh(mesh), bc, ti)

    # err_proportion=0.01 is 1% deviation of total range of 3D shape
    threshold = render_diag * err_proportion
    xyz_found = si.as_unmasked().sample(mesh.with_dims([0, 1]), order=1).T
    err = np.sum((xyz_found - mesh.points) ** 2, axis=1)

    visible = err < threshold
    return visible


def render_hi_res_shape_image(mesh, render_width=3000):
    from menpo3d.rasterize import GLRasterizer
    h, w = mesh.range()[:2]
    aspect_ratio = w / h
    height = render_width * aspect_ratio

    r = GLRasterizer(
        projection_matrix=model_to_clip_transform(mesh).h_matrix,
        width=render_width, height=height)
    return r.model_to_image_transform, r.rasterize_mesh_with_shape_image(mesh)[1]


def per_vertex_occlusion_gl_rasterizer(mesh, err_proportion=0.0001,
                                       err_norm='z', render_width=3000):
    # Render a high-resolution shape image for visibility testing

    # z scale can be very large for high focal lengths.
    # ensure z is scaled to match x/y for the purposes of masking.
    print('scaling down ')
    [x_r, y_r, z_r] = mesh.range()
    av_xy_r = (x_r + y_r) / 2.0
    mesh = Scale([1, 1, av_xy_r / z_r]).apply(mesh)

    model_to_image_transform, shape_image = render_hi_res_shape_image(mesh, render_width=render_width)
    # err_proportion=0.01 is 1% deviation of total range of 3D shape
    err_scale = mesh.range()[2].sum() if err_norm == 'z' else np.sqrt(
        (mesh.range() ** 2).sum())
    threshold = err_scale * err_proportion

    sample_points_3d = mesh
    sample_points_2d = model_to_image_transform.apply(sample_points_3d)

    xyz_found = shape_image.as_unmasked().sample(sample_points_2d, order=1).T
    err = np.sum((xyz_found - sample_points_3d.points) ** 2, axis=1)
    return err < threshold


def per_vertex_occlusion_accurate(mesh):
    from menpo3d.vtkutils import trimesh_to_vtk
    import vtk
    tol = mesh.mean_edge_length() / 1000
    min_, max_ = mesh.bounds()
    z_min = min_[-1] - 10
    z_max = max_[-1] + 10

    ray_start = mesh.points.copy()
    ray_end = mesh.points.copy()
    points = mesh.points
    ray_start[:, 2] = z_min
    ray_end[:, 2] = z_max

    vtk_mesh = trimesh_to_vtk(mesh)

    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(vtk_mesh)
    obbTree.BuildLocator()

    vtk_points = vtk.vtkPoints()
    vtk_cellIds = vtk.vtkIdList()
    bad_val = tuple(ray_start[0])
    first_intersects = []
    for start, end, point in zip(ray_start, ray_end, points):
        start = tuple(start)
        end = tuple(end)
        obbTree.IntersectWithLine(start, end, vtk_points, vtk_cellIds)
        data = vtk_points.GetData()
        break
    for start, end, point in zip(ray_start, ray_end, points):
        start = tuple(start)
        end = tuple(end)
        obbTree.IntersectWithLine(start, end, vtk_points, vtk_cellIds)
        data = vtk_points.GetData()
        if data.GetNumberOfTuples() > 0:
            first_intersects.append(data.GetTuple3(0))
        else:
            first_intersects.append(bad_val)

    visible = np.linalg.norm(points - np.array(first_intersects), axis=1) < tol
    return visible


def extract_per_vertex_colour(mesh, image):
    return image.sample(PointCloud(mesh.points[:, :2])).T


def extract_per_vertex_colour_with_occlusion(mesh, image,
                                             err_proportion=0.0001,
                                             render_diag=600):
    colours = extract_per_vertex_colour(mesh, image)
    mask = per_vertex_occlusion(mesh,
                                err_proportion=err_proportion,
                                render_diag=render_diag)
    return colours, mask


def extract_per_vertex_features(mesh, image, feature_f, diagonal_range=None):
    image = image.copy()
    image.landmarks['mesh_2d'] = mesh.with_dims([0, 1])
    if diagonal_range is not None:
        image = image.rescale_landmarks_to_diagonal_range(diagonal_range,
                                                          group='mesh_2d')
    feature_image = feature_f(image)
    return extract_per_vertex_colour(feature_image.landmarks['mesh_2d'],
                                     feature_image)
