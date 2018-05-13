import numpy as np
from menpo.shape import PointCloud
from menpo3d.camera import PerspectiveCamera


def initialize_camera(image, lms_3d, focal_length=None):
    return PerspectiveCamera.init_from_2d_projected_shape(
        lms_3d, image.landmarks[None], image.shape, focal_length=focal_length)


def initialize_camera_from_params(image, mm, id_ind, exp_ind,
                                  p, q, focal_length=None):
    shape_params = np.zeros(mm.shape_model.n_active_components)
    shape_params[id_ind] = p
    shape_params[exp_ind] = q
    mesh = mm.shape_model.instance(shape_params, normalized_weights=True)
    lms = PointCloud(mesh.points[mm.model_landmarks_index])
    return initialize_camera(image, lms, focal_length=focal_length)
