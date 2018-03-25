from menpo3d.result import (ParametricIterativeResult,
                            MultiScaleParametricIterativeResult)


class MMAlgorithmResult(ParametricIterativeResult):
    r"""
    Class for storing the iterative result of a MM optimisation algorithm.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial mesh** using the shape model. The
              generated reconstructed mesh is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    shape_parameters : `list` of `ndarray`
        The `list` of shape parameters per iteration. Note that the list
        includes the parameters of the projection of the initial mesh. The last
        member of the list corresponds to the final mesh's parameters. It must
        have the same length as `camera_parameters` and `meshes`.
    texture_parameters : `list` of `ndarray`
        The `list` of texture parameters per iteration. Note that the list
        includes the parameters of the projection of the initial mesh. The last
        member of the list corresponds to the final mesh's parameters. It must
        have the same length as `camera_transforms` and `meshes`.
    meshes : `list` of `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        The `list` of meshes per iteration. Note that the list does not
        include the initial mesh. However, it includes the reconstruction of
        the initial mesh. The last member of the list is the final mesh.
    camera_transforms : `list` of `menpo3d.camera.PerspectiveCamera`
        The `list` of camera transform objects per iteration. Note that the
        list does not include the initial camera transform. The last member
        of the list is the final camera transform.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    initial_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The initial mesh from which the fitting process started. If
        ``None``, then no initial mesh is assigned.
    initial_camera_transform : `menpo3d.camera.PerspectiveCamera` or ``None``, optional
        The initial camera transform. If ``None``, then no initial camera
        is assigned.
    gt_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The ground truth mesh associated with the image. If ``None``, then
        no ground truth mesh is assigned.
    costs : `list` of `float` or ``None``, optional
        The `list` of cost per iteration. If ``None``, then it is assumed
        that the cost function cannot be computed for the specific
        algorithm. It must have the same length as `meshes`.
    """
    def __init__(self, shape_parameters, texture_parameters, meshes,
                 camera_transforms, image=None, initial_mesh=None,
                 initial_camera_transform=None, gt_mesh=None, costs=None):
        super(MMAlgorithmResult, self).__init__(
            shape_parameters=shape_parameters, meshes=meshes,
            camera_transforms=camera_transforms, image=image,
            initial_mesh=initial_mesh,
            initial_camera_transform=initial_camera_transform, gt_mesh=gt_mesh,
            costs=costs)
        self._texture_parameters = texture_parameters

    @property
    def texture_parameters(self):
        r"""
        Returns the `list` of texture parameters obtained at each iteration
        of the fitting process. The `list` includes the parameters of the
        `initial_mesh` (if it exists) and `final_mesh`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._texture_parameters


class MMResult(MultiScaleParametricIterativeResult):
    r"""
    Class for storing the multi-scale iterative fitting result of a MM. It
    holds the meshes, shape parameters, texture parameters and costs per
    iteration.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial mesh** using the shape model. The
              generated reconstructed mesh is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    results : `list` of :map:`ParametricIterativeResult`
        The `list` of non parametric iterative results per scale.
    n_scales : `int`
        The number of scales.
    affine_transforms : `list` of `menpo.transform.Affine`
        The list of affine transforms per scale that transform where used
        dutring the image's pre-processing.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The ground truth mesh associated with the image. If ``None``, then no
        ground truth mesh is assigned.
    model_landmarks_index : `list` or ``None``, optional
        It `list`, then it is supposed to provide indices for vertices of the
        model that have some kind of semantic meaning. These points will be
        used in order to generate 2D pointclouds projected in the image plane.
        If ``None``, then the 2D pointclouds will not be generated.
    """
    def __init__(self, results, affine_transforms, n_scales, image=None,
                 gt_mesh=None, model_landmarks_index=None):
        super(MMResult, self).__init__(
            results=results, affine_transforms=affine_transforms,
            n_scales=n_scales, image=image, gt_mesh=gt_mesh,
            model_landmarks_index=model_landmarks_index)
        # Create texture parameters list
        self._texture_parameters = None
        if results[0].texture_parameters is not None:
            self._texture_parameters = []
            for r in results:
                self._texture_parameters += r.texture_parameters

    @property
    def texture_parameters(self):
        r"""
        Returns the `list` of texture parameters obtained at each iteration
        of the fitting process. The `list` includes the parameters of the
        `initial_mesh` (if it exists) and `final_mesh`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._texture_parameters
