import numpy as np

from .derivatives import (d_camera_d_shape_parameters,
                          d_camera_d_camera_parameters)
from .projectout import project_out, sample_uv_terms


def J_data(camera, warped_uv, shape_pc_uv, U_tex_pc, grad_x_uv,
           grad_y_uv, focal_length_update=False):
    # Compute derivative of camera wrt shape and camera parameters
    dp_da_dr = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)

    dp_dr = d_camera_d_camera_parameters(
        camera, warped_uv, with_focal_length=focal_length_update)

    # stack the shape_parameters/camera_parameters updates
    dp_da_dr = np.hstack((dp_da_dr, dp_dr))

    # Multiply image gradient with camera derivative
    permuted_grad_x = np.transpose(grad_x_uv[..., None], (0, 2, 1))
    permuted_grad_y = np.transpose(grad_y_uv[..., None], (0, 2, 1))
    J = permuted_grad_x * dp_da_dr[0] + permuted_grad_y * dp_da_dr[1]

    # Project-out
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    return project_out(J, U_tex_pc)


def J_lms(camera, warped_uv, shape_pc_uv, focal_length_update=False):
    # Compute derivative of camera wrt shape and camera parameters
    J = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
    dp_dr = d_camera_d_camera_parameters(
        camera, warped_uv, with_focal_length=focal_length_update)
    J = np.hstack((J, dp_dr))

    # Reshape to : n_params x (2 * N)
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    return J


def jacobians(s, c, image, lms_points_xy, mm, id_ind, exp_ind, template_camera,
              grad_x, grad_y, shape_pc, shape_pc_lms, n_samples,
              compute_costs=False):

    instance = mm.shape_model.instance(s, normalized_weights=True)
    camera = template_camera.from_vector(c)

    (instance_w, instance_in_image, warped_uv, img_error_uv,
     shape_pc_uv, U_tex_pc, grad_x_uv, grad_y_uv
     ) = sample_uv_terms(instance, image, camera, mm, shape_pc, grad_x, grad_y,
                         n_samples)

    # Data term Jacobian
    JT = J_data(camera, warped_uv, shape_pc_uv, U_tex_pc, grad_x_uv, grad_y_uv)

    # Landmarks term Jacobian
    # Get projected instance on landmarks and error term
    warped_lms = instance_in_image.points[mm.model_landmarks_index]
    lms_error_xy = (lms_points_xy - warped_lms[:, [1, 0]]).T.ravel()
    warped_view_lms = instance_w[mm.model_landmarks_index]
    J_lT = J_lms(camera, warped_view_lms, shape_pc_lms)

    # form the main two Jacobians...
    J_f = JT.T
    J_l = J_lT.T

    # ...and then slice at the appropriate indices to break down by param type.
    c_offset = id_ind.shape[0] + exp_ind.shape[0]
    jacs = {
        'J_f_p': J_f[:, id_ind],
        'J_f_q': J_f[:, exp_ind],
        'J_f_c': J_f[:, c_offset:],

        'J_l_p': J_l[:, id_ind],
        'J_l_q': J_l[:, exp_ind],
        'J_l_c': J_l[:, c_offset:],

        'e_f': img_error_uv,
        'e_l': lms_error_xy
    }

    if compute_costs:
        resid_f = project_out(img_error_uv, U_tex_pc)
        err_f = (resid_f ** 2).sum()

        resid_l = lms_error_xy
        err_l = (resid_l ** 2).sum()

        jacs['costs'] = {
            'err_f': err_f,
            'err_l': err_l
        }

    return jacs
