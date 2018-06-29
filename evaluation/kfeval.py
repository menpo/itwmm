# Extract key landmarks and build a per-vertex inclusion mask
# (as we only want to evaluate on the inner region of the face)
import numpy as np

from menpo.landmark import face_ibug_49_to_face_ibug_49
from menpo.shape import PointCloud
from menpo.transform import AlignmentSimilarity
from menpo3d.vtkutils import VTKClosestPointLocator, trimesh_to_vtk


def landmark_and_mask_gt_mesh(gt_mesh, distance=1):
    gt_mesh.landmarks['ibug49'] = face_ibug_49_to_face_ibug_49(
        gt_mesh.landmarks['ibug49'])
    gt_mesh.landmarks['nosetip'] = PointCloud(
        gt_mesh.landmarks['ibug49'].get_label('nose').points[-5][None, :])
    gt_mesh.landmarks['eye_corners'] = PointCloud(
        gt_mesh.landmarks['ibug49'].points[[36 - 17, 45 - 17]])
    eval_mask = gt_mesh.distance_to(
        gt_mesh.landmarks['nosetip']).ravel() < distance
    return gt_mesh, eval_mask


# Dense procrustes align before we compute any error
def align_dense_fit_to_gt(fit_3d, gt_mesh):
    return AlignmentSimilarity(fit_3d, gt_mesh).apply(fit_3d)


def calculate_dense_error(fit_3d_aligned, gt_mesh):
    fit_vtk = trimesh_to_vtk(fit_3d_aligned)
    closest_points_on_fit = VTKClosestPointLocator(fit_vtk)
    # for each vertex on the ground truth, find the nearest distance
    # to the aligned fit.
    nearest_points, tri_indices = closest_points_on_fit(gt_mesh.points)
    err_per_vertex = np.sqrt(np.sum((nearest_points -
                                     gt_mesh.points) ** 2, axis=1))

    # normalize by inter-oc
    b, a = gt_mesh.landmarks['eye_corners'].points
    inter_occ_distance = np.sqrt(((a - b) ** 2).sum())
    # print('norm: {}'.format(inter_occ_distance))
    return err_per_vertex / inter_occ_distance


def mask_align_and_calculate_dense_error(fit_3d, gt_mesh, mask):
    fit_3d_masked = fit_3d.from_mask(mask)
    gt_mesh_masked = gt_mesh.from_mask(mask)

    fit_3d_masked_aligned = align_dense_fit_to_gt(
        fit_3d_masked,
        gt_mesh_masked
    )

    return (calculate_dense_error(fit_3d_masked_aligned, gt_mesh_masked),
            fit_3d_masked_aligned, gt_mesh_masked)
