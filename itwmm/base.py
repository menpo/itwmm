import numpy as np

from menpo.base import copy_landmarks_and_path
from menpo.shape import ColouredTriMesh


def as_colouredtrimesh(self, colours=None, copy=True):
    """
    Converts this to a :map:`ColouredTriMesh`.

    Parameters
    ----------
    colours : ``(N, 3)`` `ndarray`, optional
        The floating point RGB colour per vertex. If not given, grey will be
        assigned to each vertex.
    copy : `bool`, optional
        If ``True``, the graph will be a copy.

    Returns
    -------
    coloured : :map:`ColouredTriMesh`
        A version of this mesh with per-vertex colour assigned.
    """
    ctm = ColouredTriMesh(self.points, trilist=self.trilist,
                          colours=colours, copy=copy)
    return copy_landmarks_and_path(self, ctm)


def instance_for_params(mm, id_ind, exp_ind, template_camera, p, q,
                        c):
    shape_params = np.zeros(mm.shape_model.n_active_components)
    shape_params[id_ind] = p
    shape_params[exp_ind] = q
    instance = mm.shape_model.instance(shape_params, normalized_weights=True)
    camera = template_camera.from_vector(c)
    instance_in_img = camera.apply(instance)
    return {
        'instance': instance,
        'instance_in_img': instance_in_img,
        'camera': camera
    }
