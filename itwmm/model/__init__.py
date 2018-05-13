import numpy as np

from menpo.model import PCAVectorModel
from menpo.visualize import print_progress

from itwmm.base import instance_for_params

from .extractimage import extract_per_vertex_colour_with_occlusion
from .math import rpca_missing


def generate_texture_model_from_itwmm(images, mm, id_ind, exp_ind,
                                      template_camera, p, qs, cs,
                                      lambda_=0.01,
                                      n_components=0.99):
    r"""Build a new texture model from an existing model and fitting
    information to a collection of images."""
    n_channels = images[0].n_channels
    n_features = n_channels * mm.n_vertices
    n_samples = len(images)
    X = np.empty((n_samples, n_features), dtype=mm.texture_model.mean().dtype)
    M = np.empty_like(X, dtype=np.bool)
    proportion_masked = []
    for i, (img, q, c) in enumerate(zip(print_progress(images, 'Extracting '
                                                               'masks and '
                                                               'features'), qs,
                                        cs)):
        i_in_img = instance_for_params(mm, id_ind, exp_ind,
                                       template_camera,
                                       p, q, c)['instance_in_img']
        features, mask = extract_per_vertex_colour_with_occlusion(i_in_img, img)
        mask_repeated = np.repeat(mask.ravel(), n_channels)
        X[i] = features.ravel()
        M[i] = mask_repeated.ravel()
        proportion_masked.append(mask.sum() / mask.shape[0])
    print('Extraction concluded. Self-occlusions on average masked {:.0%} of '
          'vertices.'.format(np.array(proportion_masked).mean()))
    print('Performing R-PCA to complete missing textures')
    A, E = rpca_missing(X, M, verbose=True, lambda_=lambda_)
    print('R-PCA completed. Building PCA model of features on completed '
          'samples.')
    model = PCAVectorModel(A, inplace=True)
    print('Trimming the components to retain only what was required.')
    model.trim_components(n_components=n_components)
    return model, X, M


def generate_texture_model_from_image_3d_fits(images_and_fits,
                                              lambda_=0.01, n_components=0.99):
    """Build an ITW texture model from a list of images with associated dense
    3D fits (one per image). Note that the input images should already have an
    image feature taken on them, and have all been resized to a consistent
    scale."""
    feat_img, fit_2d = images_and_fits[0]
    n_channels = feat_img.n_channels
    n_features = n_channels * fit_2d.n_points
    n_samples = len(images_and_fits)
    X = np.empty((n_samples, n_features), dtype=feat_img.pixels.dtype)
    M = np.empty_like(X, dtype=np.bool)
    proportion_masked = []
    for i, (img, fit_2d) in enumerate(print_progress(
            images_and_fits, prefix='Extracting masks & features')):
        features, mask = extract_per_vertex_colour_with_occlusion(fit_2d, img)
        mask_repeated = np.repeat(mask.ravel(), n_channels)
        X[i] = features.ravel()
        M[i] = mask_repeated.ravel()
        proportion_masked.append(mask.sum() / mask.shape[0])
    print('Performing R-PCA to complete missing textures')
    A, E = rpca_missing(X, M, verbose=True, lambda_=lambda_)
    print('R-PCA completed. Building PCA model of features on completed '
          'samples.')
    model = PCAVectorModel(A, inplace=True)
    print('Trimming the components to retain only what was required.')
    model.trim_components(n_components=n_components)

    return model, X, M
