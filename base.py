import numpy as np

from menpo.base import name_of_callable, Copyable
from menpo.shape import ColouredTriMesh, TexturedTriMesh, PointCloud


class MorphableModel(Copyable):
    r"""
    Abstract class for defining a Morphable Model.

    There are two subclasses, analogously to the :map:`TexturedTriMesh` and
    :map:`ColouredTriMesh`. The only differences come in two areas:

        1. How the shape and texture PCA models are combined in synthesising a
           new instance
        2. How the texture is sampled at locations in forming the cost
           function

    The vast majority of the code is shared in this superclass.

    Please see the references for a basic list of relevant papers.

    Parameters
    ----------
    shape_model : `menpo.model.PCAModel`
        The PCA model of the 3D shape. It is assumed that a shape instance is
        defined as a `menpo.shape.TriMesh`.
    texture_model : `menpo.model.PCAVectorModel`
        The PCA texture model. Note that the texture model can be feature
        based, in which case you need to specify the employed
        `holistic_features`.
    landmarks : `menpo.shape.PointUndirectedGraph`
        The set of sparse landmarks defined in the 3D space.
    holistic_features : `function`
        The features that were used for building the texture model. Please
        refer to `menpo.feature` for a list of potential features.
    diagonal : `int`
        This parameter was used to rescale the training images so that the
        diagonal of their bounding boxes matches the provided value. In other
        words, this parameter defined the size of the training images before
        extracting features.

    References
    ----------
    .. [1] V. Blanz, T. Vetter. "A morphable model for the synthesis of 3D
        faces", Conference on Computer Graphics and Interactive Techniques,
        pp. 187-194, 1999.
    .. [2] P. Paysan, R. Knothe, B. Amberg, S. Romdhani, T. Vetter. "A 3D
        face model for pose and illumination invariant face recognition",
        IEEE International Conference on Advanced Video and Signal Based
        Surveillance, pp. 296-301, 2009.
    """
    def __init__(self, shape_model, texture_model, landmarks,
                 holistic_features, diagonal):
        self.shape_model = shape_model
        self.texture_model = texture_model
        self.holistic_features = holistic_features
        self.diagonal = diagonal
        # Find mapping that brings landmarks in correspondence with shape model
        (self.model_landmarks_index,
         self.landmarks) = find_correspondences_between_shapes(
            landmarks, self.shape_model.mean(), return_pointcloud=True)

    @property
    def n_vertices(self):
        """
        Returns the number of vertices of the shape model's trimesh.

        :type: `int`
        """
        return self.shape_model.template_instance.n_points

    @property
    def n_triangles(self):
        """
        Returns the number of triangles of the shape model's trimesh.

        :type: `int`
        """
        return self.shape_model.template_instance.n_tris

    def instance(self, shape_weights=None, texture_weights=None,
                 landmark_group='landmarks'):
        r"""
        Generates a novel Morphable Model instance given a set of shape and
        texture weights. If no weights are provided, then the mean Morphable
        Model instance is returned.

        Note that the texture generated is always clipped to the range(0-1).

        Parameters
        ----------
        shape_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the shape model that will be used to create a novel
            shape instance. If ``None``, the weights are assumed to be zero,
            thus the mean shape is used.
        texture_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the texture model that will be used to create a
            novel texture instance. If ``None``, the weights are assumed
            to be zero, thus the mean appearance is used.
        landmark_group : `str`, optional
            The group name that will be used for the sparse landmarks that
            will be attached to the returned instance.

        Returns
        -------
        instance : `menpo.shape.ColouredTriMesh`
            The coloured trimesh instance.
        """
        if shape_weights is None:
            shape_weights = np.zeros(self.shape_model.n_active_components)
        if texture_weights is None:
            texture_weights = np.zeros(self.texture_model.n_active_components)

        # Generate instance
        shape_instance = self.shape_model.instance(shape_weights)
        texture_instance = self.texture_model.instance(texture_weights)

        # Create and return trimesh
        return self._instance(shape_instance, texture_instance, landmark_group)

    def random_instance(self, landmark_group='__landmarks__'):
        r"""
        Generates a random instance of the Morphable Model.

        Parameters
        ----------
        landmark_group : `str`, optional
            The group name that will be used for the sparse landmarks that
            will be attached to the returned instance. Default is
            ``'__landmarks__'``.

        Returns
        -------
        instance : `menpo.shape.ColouredTriMesh`
            The coloured trimesh instance.
        """
        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = np.random.randn(self.shape_model.n_active_components)
        shape_instance = self.shape_model.instance(shape_weights)
        texture_weights = np.random.randn(
            self.texture_model.n_active_components)
        texture_instance = self.texture_model.instance(texture_weights)

        return self._instance(shape_instance, texture_instance, landmark_group)

    def view_shape_model_widget(self, n_parameters=5,
                                parameters_bounds=(-3.0, 3.0),
                                mode='multiple'):
        r"""
        Visualizes the shape model of the Morphable Model using an interactive
        widget.

        Parameters
        ----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        """
        try:
            from menpowidgets import visualize_shape_model_3d
            visualize_shape_model_3d(self.shape_model,
                                     n_parameters=n_parameters,
                                     parameters_bounds=parameters_bounds,
                                     mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_mm_widget(self, n_shape_parameters=5, n_texture_parameters=5,
                       parameters_bounds=(-3.0, 3.0), mode='multiple'):
        r"""
        Visualizes the Morphable Model using an interactive widget.

        Parameters
        ----------
        n_shape_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        n_texture_parameters : `int` or `list` of `int` or ``None``, optional
            The number of texture principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        """
        try:
            from menpowidgets import visualize_morphable_model
            visualize_morphable_model(
                self, n_shape_parameters=n_shape_parameters,
                n_texture_parameters=n_texture_parameters,
                parameters_bounds=parameters_bounds, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        cls_str = r"""{}
 - Shape model class: {}
   - {} vertices, {} triangles
   - {} shape components
   - Instance class: {}
 - Texture model class: {}
   - {} texture components
   - Diagonal of {} pixels
   - Features function is {}
   - {} channels
 - Sparse landmarks class: {}
   - {} landmarks
""".format(self._str_title, name_of_callable(self.shape_model),
           self.n_vertices, self.n_triangles, self.shape_model.n_components,
           name_of_callable(self.shape_model.template_instance),
           name_of_callable(self.texture_model),
           self.texture_model.n_components, self.diagonal,
           name_of_callable(self.holistic_features), self.n_channels,
           name_of_callable(self.landmarks), self.landmarks.n_points)
        return cls_str


class ColouredMorphableModel(MorphableModel):
    r"""
    A Morphable Model where the texture information is assigned as a per-vertex
    Color vector.

    Parameters
    ----------
    shape_model : `menpo.model.PCAModel`
        The PCA model of the 3D shape. It is assumed that a shape instance is
        defined as a `menpo.shape.TriMesh`.
    texture_model : `menpo.model.PCAVectorModel`
        The PCA model of the per-vertex texture. It is assumed that a texture
        instance is defined as an ``(n_vertices * n_channels,)`` vector,
        where `n_vertices` should be the same as in the case of the
        `shape_model`. Note that the texture model can be feature based, in
        which case you need to specify the employed `holistic_features`.
    landmarks : `menpo.shape.PointUndirectedGraph`
        The set of sparse landmarks defined in the 3D space.
    holistic_features : `function`
        The features that were used for building the texture model. Please
        refer to `menpo.feature` for a list of potential features.
    diagonal : `int`
        This parameter was used to rescale the training images so that the
        diagonal of their bounding boxes matches the provided value. In other
        words, this parameter defined the size of the training images before
        extracting features.

    References
    ----------
    .. [1] V. Blanz, T. Vetter. "A morphable model for the synthesis of 3D
        faces", Conference on Computer Graphics and Interactive Techniques,
        pp. 187-194, 1999.
    .. [2] P. Paysan, R. Knothe, B. Amberg, S. Romdhani, T. Vetter. "A 3D
        face model for pose and illumination invariant face recognition",
        IEEE International Conference on Advanced Video and Signal Based
        Surveillance, pp. 296-301, 2009.
    """
    @property
    def _str_title(self):
        return 'Coloured Morphable Model'

    @property
    def n_channels(self):
        """
        Returns the number of channels of the texture model.

        :type: `int`
        """
        return int(self.texture_model.n_features / self.n_vertices)

    def _instance(self, shape_instance, texture_instance, landmark_group):
        # Reshape the texture instance
        texture_instance = texture_instance.reshape([-1, self.n_channels])

        # restrict the texture to 0-1
        # texture_instance = np.clip(texture_instance, 0, 1)

        # Create trimesh
        trimesh = ColouredTriMesh(shape_instance.points,
                                  trilist=shape_instance.trilist,
                                  colours=texture_instance)
        # Attach landmarks to trimesh
        trimesh.landmarks[landmark_group] = self.landmarks
        # Return trimesh
        return trimesh

    def sample_texture_model(self, bcoords, tri_indices):
        shape_template = self.shape_model.template_instance
        vertex_indices = shape_template.trilist[tri_indices]

        t_model = self.texture_model.components.reshape(
            [self.texture_model.n_active_components, -1, self.n_channels])
        # n: components    s: samples    t: triangle    c: channels
        return np.einsum('nstc, st -> scn',
                         t_model[:, vertex_indices], bcoords)

    def project_instance_on_texture_model(self, instance):
        return self.texture_model.project(instance.colours.ravel())


class TexturedMorphableModel(MorphableModel):
    r"""
    A Morphable Model where the texture information is assigned as image, with
    texture coordinates linking the vertex locations to the texture. This allows
    for a Model with differing spatial and texture resolutions.

    Parameters
    ----------
    shape_model : `menpo.model.PCAModel`
        The PCA model of the 3D shape. It is assumed that a shape instance is
        defined as a :map:`menpo.shape.TriMesh`.
    texture_model : :map:`PCAModel`.
        The PCA model of texture. It is assumed that a texture instance is
        defined as a `menpo.image.Image`. Note that the texture model can be
        feature based, in which case you need to specify the employed
        `holistic_features`.
    landmarks : `menpo.shape.PointUndirectedGraph`
        The set of sparse landmarks defined in the 3D space.
    tcoords : `menpo.shape.PointCloud`
        The texture coordinates linking any instance of the `texture_model`
        to any instance of the `shape_model`.
    holistic_features : `function`
        The features that were used for building the texture model. Please
        refer to `menpo.feature` for a list of potential features.
    diagonal : `int`
        This parameter was used to rescale the training images so that the
        diagonal of their bounding boxes matches the provided value. In other
        words, this parameter defined the size of the training images before
        extracting features.
    """
    def __init__(self, shape_model, texture_model, landmarks, tcoords,
                 holistic_features, diagonal):
        super(TexturedMorphableModel, self).__init__(
            shape_model, texture_model, landmarks, holistic_features, diagonal)
        self.tcoords = tcoords
        self.tcoords_pixel_scaled = self.instance().tcoords_pixel_scaled()

    @property
    def _str_title(self):
        return 'Textured Morphable Model'

    @property
    def n_channels(self):
        """
        Returns the number of channels of the texture model.

        :type: `int`
        """
        return self.texture_model.template_instance.n_channels

    def _instance(self, shape_instance, texture_instance, landmark_group):
        # Create trimesh

        # restrict the texture to 0-1
        # texture_instance.pixels = np.clip(texture_instance.pixels, 0, 1)

        trimesh = TexturedTriMesh(shape_instance.points,
                                  trilist=shape_instance.trilist,
                                  tcoords=self.tcoords.points,
                                  texture=texture_instance)
        # Attach landmarks to trimesh
        trimesh.landmarks[landmark_group] = self.landmarks
        # Return trimesh
        return trimesh

    def sample_texture_model(self, bcoords, tri_indices):
        shape_template = self.shape_model.template_instance
        texture_template = self.texture_model.template_instance

        sample_points_in_texture = shape_template.barycentric_coordinate_interpolation(
            self.tcoords_pixel_scaled.points, bcoords, tri_indices)

        to_index = sample_points_in_texture.round().astype(int)

        texture_index_img = texture_template.from_vector(
            np.arange(self.texture_model.n_features))

        # TODO shouldn't get out of mask samples here, but we do (hence
        # unmasked). This means for some samples every time in UV space
        # we are getting the wrong point.
        pca_sample_locations = texture_index_img.as_unmasked().sample(to_index, order=0)

        x = self.texture_model.components[:, pca_sample_locations]
        return np.swapaxes(x, 0, 2)

    def project_instance_on_texture_model(self, instance):
        return self.texture_model.project(instance.texture)


def find_correspondences_between_shapes(source, target, return_pointcloud=True):
    """
    It maps the landmarks in the source pointcloud to their corresponding
    indices in the target pointcloud. It optionally returns the source
    pointcloud with landmarks in correspondence to the target pointcloud.

    Parameters
    ----------
    source : `menpo.shape.PointCloud`
        The source pointcloud.
    target : `menpo.shape.PointCloud`
        The target pointcloud.
    return_pointcloud : `bool`, optional
        If ``True``, then the source pointcloud with landmarks in correspondence
        to the target pointcloud. It is returned only
        if ``return_pointcloud == True``.

    Returns
    -------
    map_source_to_target : ``(source_n_points,)`` `ndarray`
        The mapping between the points of source and target.
    source_in_correspondence : `menpo.shape.PointCloud`
        The source pointcloud with landmarks in correspondence will also be
        returned.
    """
    # Compute distance between source and target landmarks
    distances = source.distance_to(target)

    # Map source landmarks to target landmarks
    map_source_to_target = np.argmin(distances, 1)

    # Create pointcloud with source landmarks that are in correspondence
    # with target
    if return_pointcloud:
        return map_source_to_target, PointCloud(
            target.points[map_source_to_target, :])
    else:
        return map_source_to_target
