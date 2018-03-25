import numpy as np

from menpo3d.camera import PerspectiveCamera, OrthographicCamera


def d_perspective_camera_d_shape_parameters(shape_pc_uv, warped_uv, camera):
    """
    Calculates the derivative of the perspective projection with respect to the
    shape parameters.

    Parameters
    ----------
    shape_pc_uv : ``(n_points, 3, n_parameters)`` `ndarray`
        The (sampled) basis of the shape model.
    warped_uv : ``(n_points, 3)`` `ndarray`
        The shape instance with the view transform (rotation and translation)
        applied on it.
    camera : `menpo3d.camera.PerspectiveCamera`
        The camera object that is responsible of projecting the model to the
        image plane.

    Returns
    -------
    dw_da : ``(2, n_shape_parameters, n_points)`` `ndarray`
        The derivative of the perspective camera transform with respect to
        the shape parameters.
    """
    n_points, n_dims, n_parameters = shape_pc_uv.shape
    assert n_dims == 3

    # Compute constant
    # (focal length divided by squared Z dimension of warped shape)
    z = warped_uv[:, 2]

    # n_dims, n_parameters, n_points
    dw_da = camera.rotation_transform.apply(shape_pc_uv.transpose(0, 2, 1)).T

    dw_da[:2] -= warped_uv[:, :2].T[:, None] * dw_da[2] / z

    return camera.projection_transform.focal_length * dw_da[:2] / z


def d_orthographic_camera_d_shape_parameters(shape_pc_uv, camera):
    """
    Calculates the derivative of the orthographic projection with respect to
    the shape parameters.

    Parameters
    ----------
    shape_pc_uv : ``(n_points, 3, n_parameters)`` `ndarray`
        The (sampled) basis of the shape model.
    camera : `menpo3d.camera.OrthographicCamera`
        The camera object that is responsible of projecting the model to the
        image plane.

    Returns
    -------
    dw_da : ``(2, n_shape_parameters, n_points)`` `ndarray`
        The derivative of the perspective camera transform with respect to
        the shape parameters.
    """
    n_points, n_dims, n_parameters = shape_pc_uv.shape
    assert n_dims == 3

    # n_dims, n_parameters, n_points
    dp_da_uv = camera.rotation_transform.apply(shape_pc_uv.transpose(0, 2, 1)).T

    return camera.projection_transform.focal_length * dp_da_uv[:2]


def d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv):
    if camera.__class__ == PerspectiveCamera:
        dp_da = d_perspective_camera_d_shape_parameters(shape_pc_uv, warped_uv,
                                                        camera)
    elif camera.__class__ == OrthographicCamera:
        dp_da = d_orthographic_camera_d_shape_parameters(shape_pc_uv, camera)
    else:
        raise ValueError("Camera must be either Perspective or "
                         "Orthographic.")
    return dp_da


def d_perspective_camera_d_camera_parameters(warped_uv, camera,
                                             with_focal_length=True,
                                             with_first_quaternion=False):
    """
    Calculates the derivative of the perspective projection with respect to the
    camera parameters.

    Parameters
    ----------
    warped_uv : ``(n_points, 3)`` `ndarray`
        The shape instance with the view transform (rotation and translation)
        applied on it.
    camera : `menpo3d.camera.PerspectiveCamera`
        The camera object that is responsible of projecting the model to the
        image plane.
    with_focal_length : `bool`, optional
        If ``False``, then the derivative with respect to the focal length
        parameter won't be returned.
    with_first_quaternion : `bool`, optional
        If ``False``, then the derivative with respect to the first
        quaternion parameter won't be returned.

    Returns
    -------
    dw_dr : ``(2, n_parameters, n_points)`` `ndarray`
        The derivative of the perspective camera transform with respect to
        the camera parameters.
    """
    n_points, n_dims = warped_uv.shape
    assert n_dims == 3

    # Find total number of parameters
    n_parameters = camera.n_parameters
    if not with_focal_length:
        n_parameters -= 1
    if not with_first_quaternion:
        n_parameters -= 1

    # Initialize derivative
    dw_dr = np.zeros((2, n_parameters, n_points))

    # Initialize parameter counter
    r = 0

    # Get z-component of warped
    z = warped_uv[:, 2]

    # Focal length, if requested
    if with_focal_length:
        dw_dr[:, r] = (warped_uv[:, :2] / z[..., None]).T
        r += 1

    # Quaternions
    centered_warped_uv = camera.translation_transform.pseudoinverse().apply(
        warped_uv).T
    # q_1, if requested
    if with_first_quaternion:
        r0 = 2 * np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]]).dot(centered_warped_uv).T
        dw_dr[:, r] = r0[:, :2].T - r0[:, 2] * warped_uv[:, :2].T / z
        r += 1

    # q_2
    r1 = 2 * np.array([[0, 0,  0],
                       [0, 0, -1],
                       [0, 1,  0]]).dot(centered_warped_uv).T
    dw_dr[:, r] = r1[:, :2].T - r1[:, 2] * warped_uv[:, :2].T / z
    r += 1

    # q_3
    r2 = 2 * np.array([[ 0, 0, 1],
                       [ 0, 0, 0],
                       [-1, 0, 0]]).dot(centered_warped_uv).T
    dw_dr[:, r] = r2[:, :2].T - r2[:, 2] * warped_uv[:, :2].T / z
    r += 1

    # q_4
    r3 = 2 * np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 0]]).dot(centered_warped_uv).T
    dw_dr[:, r] = r3[:, :2].T - r3[:, 2] * warped_uv[:, :2].T / z
    r += 1

    # constant multiplication
    i1 = 1 if with_focal_length else 0
    i2 = i1 + 4 if with_first_quaternion else i1 + 3
    dw_dr[:, i1:i2] *= camera.projection_transform.focal_length / z

    # Translations
    # t_x
    dw_dr[0, r] = camera.projection_transform.focal_length / z
    r += 1
    # t_y
    dw_dr[1, r] = camera.projection_transform.focal_length / z
    r += 1
    # t_z
    dw_dr[:, r] = (- camera.projection_transform.focal_length *
                   warped_uv[:, :2] / z[..., None] ** 2).T

    return dw_dr


def d_orthographic_camera_d_camera_parameters(warped_uv, camera,
                                              with_focal_length=True,
                                              with_first_quaternion=False):
    """
    Calculates the derivative of the orthographic projection with respect to the
    camera parameters.

    Parameters
    ----------
    warped_uv : ``(n_points, 3)`` `ndarray`
        The shape instance with the view transform (rotation and translation)
        applied on it.
    camera : `menpo3d.camera.PerspectiveCamera`
        The camera object that is responsible of projecting the model to the
        image plane.
    with_focal_length : `bool`, optional
        If ``False``, then the derivative with respect to the focal length
        parameter won't be returned.
    with_first_quaternion : `bool`, optional
        If ``False``, then the derivative with respect to the first
        quaternion parameter won't be returned.

    Returns
    -------
    dw_dr : ``(2, n_parameters, n_points)`` `ndarray`
        The derivative of the orthographic camera transform with respect to
        the camera parameters.
    """
    n_points, n_dims = warped_uv.shape
    assert n_dims == 3

    # Find total number of parameters
    n_parameters = camera.n_parameters
    if not with_focal_length:
        n_parameters -= 1
    if not with_first_quaternion:
        n_parameters -= 1

    # Initialize derivative
    dw_dr = np.zeros((2, n_parameters, n_points))

    # Initialize parameter counter
    r = 0

    # Focal length, if requested
    if with_focal_length:
        dw_dr[:, r] = warped_uv[:, :2].T
        r += 1

    # Quaternions
    centered_warped_uv = camera.translation_transform.pseudoinverse().apply(
        warped_uv).T

    # q_1, if requested
    if with_first_quaternion:
        r0 = 2 * np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]]).dot(centered_warped_uv).T
        dw_dr[:, r] = camera.projection_transform.focal_length * r0[:, :2].T
        r += 1

    # q_2
    r1 = 2 * np.array([[0, 0,  0],
                       [0, 0, -1],
                       [0, 1,  0]]).dot(centered_warped_uv).T
    dw_dr[:, r] = camera.projection_transform.focal_length * r1[:, :2].T
    r += 1

    # q_3
    r2 = 2 * np.array([[ 0, 0, 1],
                       [ 0, 0, 0],
                       [-1, 0, 0]]).dot(centered_warped_uv).T
    dw_dr[:, r] = camera.projection_transform.focal_length * r2[:, :2].T
    r += 1

    # q_4
    r3 = 2 * np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 0]]).dot(centered_warped_uv).T
    dw_dr[:, r] = camera.projection_transform.focal_length * r3[:, :2].T
    r += 1

    # Translations
    # t_x
    dw_dr[0, r] = camera.projection_transform.focal_length
    r += 1

    # t_y
    dw_dr[1, r] = camera.projection_transform.focal_length
    r += 1

    return dw_dr


def d_camera_d_camera_parameters(camera, warped_uv, with_focal_length):
    if camera.__class__ == PerspectiveCamera:
        dp_dr = d_perspective_camera_d_camera_parameters(
            warped_uv, camera, with_focal_length=with_focal_length,
            with_first_quaternion=False)
    elif camera.__class__ == OrthographicCamera:
        dp_dr = d_orthographic_camera_d_camera_parameters(
            warped_uv, camera, with_focal_length=with_focal_length,
            with_first_quaternion=False)
    else:
        raise ValueError("Camera must be either Perspective or "
                         "Orthographic.")
    return dp_dr
