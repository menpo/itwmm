import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import menpo.io as mio
from menpo.landmark import face_ibug_49_to_face_ibug_49
from menpo.shape import TriMesh, PointCloud
import menpo3d.io as m3io
from menpofit.visualize import (plot_cumulative_error_distribution,
                                statistics_table, 
                                print_progress)

from .eos import load_eos_low_res_lm_index, upsample_eos_low_res_to_fw_no_texture
from .kfeval import (landmark_and_mask_gt_mesh,
                     mask_align_and_calculate_dense_error)


THIS_DIR = Path(__file__).parent

# landmarks indexes of the LSFM model
lms_indexes = mio.import_pickle(THIS_DIR / 'lsfm_lms_indexes.pkl')
                
    
def calculate_errors_3dMDLab(path_fits, path_gt, model):
    r"""
    Parameters
    ----------
     path_fits: 'string' of the directory that contains your reconstructed
                meshes. Meshes' filenames should be the same with the
                corresponding images/ground truth meshes filenames + the suffix
                of the mesh type (e.g .obj, .ply)
     path_gt: 'string' of the directory that contains the ground truth meshes.
     model: 'string' of the model template you are using. Should be one of the
            following: 'LSFM', 'Basel', 'Surrey'
    Returns
    -------
     errors: python list that contains the error per vertex for all meshes
    """
    path_fits = Path(path_fits)
    path_gt = Path(path_gt)
    
    # load meshes' filenames
    filenames = [p.name for p in path_fits.glob('*')]
    filenames.sort()

    errors = [0]*len(filenames)
    for i, filename in enumerate(print_progress(filenames)):

        # load gt_mesh, fit_3d
        gt_mesh = m3io.import_mesh(path_gt / filename, texture_resolver=None)
        gt_mesh.landmarks['ibug49'] = PointCloud(
            gt_mesh.points[lms_indexes][19:])
        fit_3d = m3io.import_mesh(path_fits / filename, texture_resolver=None)

        if model == 'Surrey':
            lms = face_ibug_49_to_face_ibug_49(PointCloud(
                    fit_3d.points[load_eos_low_res_lm_index()]))
            fit_3d = upsample_eos_low_res_to_fw_no_texture(fit_3d)
            fit_3d.landmarks['ibug49'] = lms
        elif model == 'LSFM' or model == 'Basel':
            fit_3d.landmarks['ibug49'] = PointCloud(
                fit_3d.points[lms_indexes][19:])
        else:
            print('Error: Not supported model template')
            return

        # calculate per vertex errors
        gt_mesh, eval_mask = landmark_and_mask_gt_mesh(gt_mesh, distance=1)
        errors[i], _, _ = mask_align_and_calculate_dense_error(fit_3d, gt_mesh,
                                                               eval_mask)
    
    return errors


def calculate_errors_4DMaja_synthetic(path_fits, path_gt, model):
    r"""
    Parameters
    ----------
     path_fits: 'string' of the directory that contains your reconstructed meshes.
                Meshes' filenames should be the same with the
                corresponding ground truth meshes filenames + the suffix of
                the mesh type (e.g <frame_number>.obj)
     path_gt: 'string' of the directory that contains the ground truth meshes.
     model: 'string' of the model template you are using. Should be one of the
            following: 'LSFM', 'Basel', 'Surrey'
    Returns
    -------
     errors: python list that contains the error per vertex for all meshes
    """
    return calculate_errors_3dMDLab(path_fits, path_gt, model)


def calculate_errors_4DMaja_real(path_fits, path_gt, model):
    r"""
    Parameters
    ----------
     path_fits: 'string' of the directory that contains your reconstructed meshes
                Meshes' filenames should be the same with the
                corresponding ground truth meshes filenames + the suffix of
                the mesh type (e.g <frame_number>.obj, <frame_number>.ply)
     path_gt: 'string' of the full path of the ground truth mesh.
     model: 'string' of the model template you are using. Should be one of the
            following: 'LSFM', 'Basel', 'Surrey'
    Returns
    -------
     errors: python list that contains the error per vertex between the ground
             truth mesh and a mean mesh calculated from your reconstructed meshes
    """
    path_fits = Path(path_fits)
    path_gt = Path(path_gt)
    
    # load meshes' filenames
    filenames = sorted([p.name for p in path_fits.glob('*')])

    # load gt_mesh (it is only one, Maja's neutral face)
    gt_mesh = m3io.import_mesh(path_gt, texture_resolver=None)
    gt_mesh.landmarks['ibug49'] = PointCloud(gt_mesh.points[lms_indexes][19:])

    errors = [0]

    # accumulate fits
    acc_points = np.zeros((gt_mesh.n_points, 3))
    for i, filename in enumerate(print_progress(filenames)):

        fit_3d = m3io.import_mesh(path_fits / filename, texture_resolver=None)
        if model == 'Surrey':
            lms = face_ibug_49_to_face_ibug_49(PointCloud(
                    fit_3d.points[load_eos_low_res_lm_index()]))
            fit_3d = upsample_eos_low_res_to_fw_no_texture(fit_3d)
            fit_3d.landmarks['ibug49'] = lms
        elif model == 'LSFM' or model == 'Basel':
            fit_3d.landmarks['ibug49'] = PointCloud(
                fit_3d.points[lms_indexes][19:])
        else:
            print('Error: Not supported model template')
            return

        acc_points += fit_3d.points

    # create mean_fit_3d
    mean_fit_3d = TriMesh(acc_points / len(filenames), gt_mesh.trilist)
    mean_fit_3d.landmarks['ibug49'] = PointCloud(
        mean_fit_3d.points[lms_indexes][19:])

    # calculate per vertex errors between the neutral gt_mesh and the mean_fit_3d
    gt_mesh, eval_mask = landmark_and_mask_gt_mesh(gt_mesh, distance=1)
    errors[0], _, _ = mask_align_and_calculate_dense_error(mean_fit_3d,
                                                           gt_mesh, eval_mask)
    
    return errors
   
    
def plot_ced_3dMDLab(errors):
    
    errs = [[error_vtx for error_mesh in errors[key] for error_vtx in error_mesh] for key in errors.keys()]
    legs = [method for method in errors.keys()]
    
    tab = statistics_table(errs, legs, 0.105, 0.01, precision=4, sort_by='auc')
    
    r = plot_cumulative_error_distribution(
        errs, legend_font_size=20, line_width=5, marker_size=10,
        axes_font_size=20, error_range=[0, 0.105, 0.01], marker_edge_width=4,
        y_label='Vertexes proportion', x_label='Normalized dense vertex error',
        title='', legend_entries=legs,
        axes_y_ticks=list(np.arange(0., 1.05, 0.1)), figure_size=(7, 5))
    plt.gca().set_axisbelow(True)
    plt.gca().yaxis.grid(color='gray', linestyle='dashed')
    return tab, r


def plot_ced_4DMaja_synthetic(errors):
    return plot_ced_3dMDLab(errors)


def plot_errors_per_frame_4DMaja_synthetic(errors, method_name):
    
    errs = [error.mean() for error in errors]

    axes = plt.gca()
    axes.set_ylim([0, 0.150])
    axes.set_xlim([1, 440])
    plt.plot(errs, linewidth=2)
    plt.title('4DMaja_synthetic - Mean errors per frame - ' + method_name)

    
def plot_ced_4DMaja_real(errors):
    return plot_ced_3dMDLab(errors)
