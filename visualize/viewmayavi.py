import numpy as np

from menpo.visualize import Renderer

# The colour map used for all lines and markers
GLOBAL_CMAP = 'jet'


def _parse_marker_size(marker_size, points):
    if marker_size is None:
        from menpo.shape import PointCloud
        pc = PointCloud(points, copy=False)
        # This is the way that mayavi automatically computes the scale factor in
        # case the user passes scale_factor = 'auto'. We use it for both the
        # marker_size as well as the numbers_size.
        xyz_min, xyz_max = pc.bounds()
        x_min, y_min, z_min = xyz_min
        x_max, y_max, z_max = xyz_max
        distance = np.sqrt(((x_max - x_min) ** 2 +
                            (y_max - y_min) ** 2 +
                            (z_max - z_min) ** 2) /
                           (4 * pc.n_points ** 0.33))
        if distance == 0:
            marker_size = 1
        else:
            marker_size = 0.1 * distance
    return marker_size


def _parse_colour(colour):
    from matplotlib.colors import ColorConverter
    return ColorConverter().to_rgb(colour)


def _check_colours_list(render_flag, colours_list, n_objects, error_str):
    from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
    if render_flag:
        if colours_list is None:
            # sample colours from jet colour map
            colours_list = sample_colours_from_colourmap(n_objects, GLOBAL_CMAP)
        if isinstance(colours_list, list):
            if len(colours_list) == 1:
                colours_list[0] = _parse_colour(colours_list[0])
                colours_list *= n_objects
            elif len(colours_list) != n_objects:
                raise ValueError(error_str)
        else:
            colours_list = [_parse_colour(colours_list)] * n_objects
    else:
        colours_list = [None] * n_objects
    return colours_list


def _set_numbering(figure, centers, render_numbering=True, numbers_size=None,
                   numbers_colour='k'):
    import mayavi.mlab as mlab
    numbers_colour = _parse_colour(numbers_colour)
    numbers_size = _parse_marker_size(numbers_size, centers)
    if render_numbering:
        for k, p in enumerate(centers):
            mlab.text3d(p[0], p[1], p[2], str(k), figure=figure,
                        scale=numbers_size, orient_to_camera=True,
                        color=numbers_colour, line_width=2)


class MayaviRenderer(Renderer):
    """
    Abstract class for performing visualizations using Mayavi.

    Parameters
    ----------
    figure_id : str or `None`
        A figure name or `None`. `None` assumes we maintain the Mayavi
        state machine and use `mlab.gcf()`.
    new_figure : bool
        If `True`, creates a new figure to render on.
    """
    def __init__(self, figure_id, new_figure):
        try:
            import mayavi.mlab as mlab
        except ImportError:
            raise ImportError("mayavi is required for viewing 3D objects "
                              "(consider 'conda/pip install mayavi')")
        super(MayaviRenderer, self).__init__(figure_id, new_figure)

        self._supported_ext = ['png', 'jpg', 'bmp', 'tiff',  # 2D
                               'ps', 'eps', 'pdf',  # 2D
                               'rib', 'oogl', 'iv', 'vrml', 'obj']  # 3D
        n_ext = len(self._supported_ext)
        func_list = [lambda obj, fp, **kwargs: mlab.savefig(fp.name, **obj)] * n_ext
        self._extensions_map = dict(zip(['.' + s for s in self._supported_ext],
                                    func_list))
        # To store actors for clearing
        self._actors = []

    def get_figure(self):
        r"""
        Gets the figure specified by the combination of `self.figure_id` and
        `self.new_figure`. If `self.figure_id == None` then `mlab.gcf()`
        is used. `self.figure_id` is also set to the correct id of the figure
        if a new figure is created.

        Returns
        -------
        figure : Mayavi figure object
            The figure we will be rendering on.
        """
        import mayavi.mlab as mlab
        if self.new_figure or self.figure_id is not None:
            self.figure = mlab.figure(self.figure_id, bgcolor=(1, 1, 1))
            # and reset the view to z forward, y up.
            self.figure.scene.camera.view_up = np.array([0, 1, 0])
        else:
            self.figure = mlab.gcf()

        self.figure_id = self.figure.name

        return self.figure

    def save_figure(self, filename, format='png', size=None,
                    magnification='auto', overwrite=False):
        r"""
        Method for saving the figure of the current `figure_id` to file.

        Parameters
        ----------
        filename : `str` or `file`-like object
            The string path or file-like object to save the figure at/into.
        format : `str`
            The format to use. This must match the file path if the file path is
            a `str`.
        size : `tuple` of `int` or ``None``, optional
            The size of the image created (unless magnification is set,
            in which case it is the size of the window used for rendering). If
            ``None``, then the figure size is used.
        magnification :	`double` or ``'auto'``, optional
            The magnification is the scaling between the pixels on the screen,
            and the pixels in the file saved. If you do not specify it, it will
            be calculated so that the file is saved with the specified size.
            If you specify a magnification, Mayavi will use the given size as a
            screen size, and the file size will be ``magnification * size``.
            If ``'auto'``, then the magnification will be set automatically.
        overwrite : `bool`, optional
            If ``True``, the file will be overwritten if it already exists.
        """
        from menpo.io.output.base import _export
        savefig_args = {'size': size, 'figure': self.figure,
                        'magnification': magnification}
        # Use the export code so that we have a consistent interface
        _export(savefig_args, filename, self._extensions_map, format,
                overwrite=overwrite)

    @property
    def width(self):
        r"""
        The width of the scene in pixels.

        :type: `int`
        """
        return self.figure.scene.get_size()[0]

    @property
    def height(self):
        r"""
        The height of the scene in pixels.

        :type: `int`
        """
        return self.figure.scene.get_size()[1]

    @property
    def modelview_matrix(self):
        r"""
        Retrieves the modelview matrix for this scene.

        :type: ``(4, 4)`` `ndarray`
        """
        camera = self.figure.scene.camera
        return camera.view_transform_matrix.to_array().astype(np.float32)

    @property
    def projection_matrix(self):
        r"""
        Retrieves the projection matrix for this scene.

        :type: ``(4, 4)`` `ndarray`
        """
        scene = self.figure.scene
        camera = scene.camera
        scene_size = tuple(scene.get_size())
        aspect_ratio = float(scene_size[0]) / float(scene_size[1])
        p = camera.get_projection_transform_matrix(
            aspect_ratio, -1, 1).to_array().astype(np.float32)
        return p

    @property
    def renderer_settings(self):
        r"""
        Returns all the information required to construct an identical
        renderer to this one.

        Returns
        -------
        settings : `dict`
            The dictionary with the following keys:

                * ``'width'`` (`int`) : The width of the scene.
                * ``'height'`` (`int`) : The height of the scene.
                * ``'model_matrix'`` (`ndarray`) : The model array (identity).
                * ``'view_matrix'`` (`ndarray`) : The view array.
                * ``'projection_matrix'`` (`ndarray`) : The projection array.

        """
        return {'width': self.width,
                'height': self.height,
                'model_matrix': np.eye(4, dtype=np.float32),
                'view_matrix': self.modelview_matrix,
                'projection_matrix': self.projection_matrix}

    def clear_figure(self):
        r"""
        Method for clearing the current figure.
        """
        from mayavi import mlab
        mlab.clf(figure=self.figure)
        if len(self._actors) > 0:
            self.figure.scene.remove_actors(self._actors)

    def force_draw(self):
        r"""
        Method for forcing the current figure to render. This is useful for
        the widgets animation.
        """
        from pyface.api import GUI
        _gui = GUI()
        orig_val = _gui.busy
        _gui.set_busy(busy=True)
        _gui.set_busy(busy=orig_val)
        _gui.process_events()


class MayaviVectorViewer3d(MayaviRenderer):
    def __init__(self, figure_id, new_figure, points, vectors):
        super(MayaviVectorViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.vectors = vectors

    def render(self, colour='r', line_width=2, marker_style='2darrow',
               marker_resolution=8, marker_size=None, step=None, alpha=1.0):
        from mayavi import mlab
        marker_size = _parse_marker_size(marker_size, self.points)
        colour = _parse_colour(colour)
        mlab.quiver3d(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                      self.vectors[:, 0], self.vectors[:, 1], self.vectors[:, 2],
                      figure=self.figure, color=colour, mask_points=step,
                      line_width=line_width, mode=marker_style,
                      resolution=marker_resolution, opacity=alpha,
                      scale_factor=marker_size)
        return self


class MayaviPointGraphViewer3d(MayaviRenderer):
    def __init__(self, figure_id, new_figure, points, edges):
        super(MayaviPointGraphViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.edges = edges

    def render(self, render_lines=True, line_colour='r', line_width=2,
               render_markers=True, marker_style='sphere', marker_size=None,
               marker_colour='r', marker_resolution=8, step=None, alpha=1.0,
               render_numbering=False, numbers_colour='k', numbers_size=None):
        from mayavi import mlab

        # Render the lines if requested
        if render_lines:
            line_colour = _parse_colour(line_colour)
            # TODO: Make step work for lines as well
            # Create the points
            if step is None:
                step = 1
            src = mlab.pipeline.scalar_scatter(self.points[:, 0],
                                               self.points[:, 1],
                                               self.points[:, 2])
            # Connect them
            src.mlab_source.dataset.lines = self.edges
            # The stripper filter cleans up connected lines
            lines = mlab.pipeline.stripper(src)

            # Finally, display the set of lines
            mlab.pipeline.surface(lines, figure=self.figure, opacity=alpha,
                                  line_width=line_width, color=line_colour)

        # Render the markers if requested
        if render_markers:
            marker_size = _parse_marker_size(marker_size, self.points)
            marker_colour = _parse_colour(marker_colour)
            mlab.points3d(self.points[:, 0], self.points[:, 1],
                          self.points[:, 2], figure=self.figure,
                          scale_factor=marker_size, mode=marker_style,
                          color=marker_colour, opacity=alpha,
                          resolution=marker_resolution, mask_points=step)

        # set numbering
        _set_numbering(self.figure, self.points, numbers_size=numbers_size,
                       render_numbering=render_numbering,
                       numbers_colour=numbers_colour)

        return self


class MayaviTriMeshViewer3d(MayaviRenderer):
    def __init__(self, figure_id, new_figure, points, trilist):
        super(MayaviTriMeshViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.trilist = trilist

    def _render_mesh(self, mesh_type, line_width, colour, marker_size,
                     marker_resolution, marker_style, step, alpha):
        import mayavi.mlab as mlab
        marker_size = _parse_marker_size(marker_size, self.points)
        colour = _parse_colour(colour)
        mlab.triangular_mesh(self.points[:, 0], self.points[:, 1],
                             self.points[:, 2], self.trilist,
                             figure=self.figure, line_width=line_width,
                             representation=mesh_type, color=colour,
                             scale_factor=marker_size, mask_points=step,
                             resolution=marker_resolution, mode=marker_style,
                             opacity=alpha, tube_radius=None)

    def render(self, mesh_type='wireframe', line_width=2, colour='r',
               marker_style='sphere', marker_size=None, marker_resolution=8,
               normals=None, normals_colour='k', normals_line_width=2,
               normals_marker_style='2darrow', normals_marker_size=None,
               normals_marker_resolution=8, step=None, alpha=1.0):
        if normals is not None:
            MayaviVectorViewer3d(self.figure_id, False,
                                 self.points, normals).render(
                colour=normals_colour, line_width=normals_line_width, step=step,
                marker_style=normals_marker_style,
                marker_resolution=normals_marker_resolution,
                marker_size=normals_marker_size, alpha=alpha)
        self._render_mesh(mesh_type, line_width, colour, marker_size,
                          marker_resolution, marker_style, step, alpha)
        return self


class MayaviTexturedTriMeshViewer3d(MayaviRenderer):
    def __init__(self, figure_id, new_figure, points, trilist, texture,
                 tcoords_per_point):
        super(MayaviTexturedTriMeshViewer3d, self).__init__(figure_id,
                                                            new_figure)
        self.points = points
        self.trilist = trilist
        self.texture = texture
        self.tcoords_per_point = tcoords_per_point
        self._actors = []

    def _render_mesh(self, mesh_type='surface', ambient_light=0.0,
                     specular_light=0.0, alpha=1.0):
        from tvtk.api import tvtk
        pd = tvtk.PolyData()
        pd.points = self.points
        pd.polys = self.trilist
        pd.point_data.t_coords = self.tcoords_per_point
        mapper = tvtk.PolyDataMapper()
        mapper.set_input_data(pd)
        p = tvtk.Property(representation=mesh_type, opacity=alpha,
                          ambient=ambient_light, specular=specular_light)
        actor = tvtk.Actor(mapper=mapper, property=p)
        # Get the pixels from our image class which are [0, 1] and scale
        # back to valid pixels. Then convert to tvtk ImageData.
        texture = self.texture.pixels_with_channels_at_back(out_dtype=np.uint8)
        if self.texture.n_channels == 1:
            texture = np.stack([texture] * 3, axis=-1)
        image_data = np.flipud(texture).ravel()
        image_data = image_data.reshape([-1, 3])
        image = tvtk.ImageData()
        image.point_data.scalars = image_data
        image.dimensions = self.texture.width, self.texture.height, 1
        texture = tvtk.Texture()
        texture.set_input_data(image)
        actor.texture = texture
        self.figure.scene.add_actors(actor)
        self._actors.append(actor)

    def render(self, mesh_type='surface', ambient_light=0.0, specular_light=0.0,
               normals=None, normals_colour='k', normals_line_width=2,
               normals_marker_style='2darrow', normals_marker_resolution=8,
               normals_marker_size=None, step=None, alpha=1.0):
        if normals is not None:
            MayaviVectorViewer3d(self.figure_id, False,
                                 self.points, normals).render(
                colour=normals_colour, line_width=normals_line_width, step=step,
                marker_style=normals_marker_style,
                marker_resolution=normals_marker_resolution,
                marker_size=normals_marker_size, alpha=alpha)
        self._render_mesh(mesh_type=mesh_type, ambient_light=ambient_light,
                          specular_light=specular_light, alpha=alpha)
        return self


class MayaviColouredTriMeshViewer3d(MayaviRenderer):
    def __init__(self, figure_id, new_figure, points, trilist,
                 colour_per_point):
        super(MayaviColouredTriMeshViewer3d, self).__init__(figure_id,
                                                            new_figure)
        self.points = points
        self.trilist = trilist
        self.colour_per_point = colour_per_point
        self._actors = []

    def _render_mesh(self, mesh_type='surface', ambient_light=0.0,
                     specular_light=0.0, alpha=1.0):
        from tvtk.api import tvtk
        pd = tvtk.PolyData()
        pd.points = self.points
        pd.polys = self.trilist
        pd.point_data.scalars = (self.colour_per_point * 255.).astype(np.uint8)
        mapper = tvtk.PolyDataMapper()
        mapper.set_input_data(pd)
        p = tvtk.Property(representation=mesh_type, opacity=alpha,
                          ambient=ambient_light, specular=specular_light)
        actor = tvtk.Actor(mapper=mapper, property=p)
        self.figure.scene.add_actors(actor)
        self._actors.append(actor)

    def render(self, mesh_type='surface', ambient_light=0.0, specular_light=0.0,
               normals=None, normals_colour='k', normals_line_width=2,
               normals_marker_style='2darrow', normals_marker_resolution=8,
               normals_marker_size=None, step=None, alpha=1.0):
        if normals is not None:
            MayaviVectorViewer3d(self.figure_id, False,
                                 self.points, normals).render(
                colour=normals_colour, line_width=normals_line_width, step=step,
                marker_style=normals_marker_style,
                marker_resolution=normals_marker_resolution,
                marker_size=normals_marker_size, alpha=alpha)
        self._render_mesh(mesh_type=mesh_type, ambient_light=ambient_light,
                          specular_light=specular_light, alpha=alpha)
        return self


class MayaviSurfaceViewer3d(MayaviRenderer):
    def __init__(self, figure_id, new_figure, values, mask=None):
        super(MayaviSurfaceViewer3d, self).__init__(figure_id, new_figure)
        if mask is not None:
            values[~mask] = np.nan
        self.values = values

    def render(self, colour=(1, 0, 0), line_width=2, step=None,
               marker_style='2darrow', marker_resolution=8, marker_size=0.05,
               alpha=1.0):
        from mayavi import mlab
        warp_scale = kwargs.get('warp_scale', 'auto')
        mlab.surf(self.values, warp_scale=warp_scale)
        return self


class MayaviLandmarkViewer3d(MayaviRenderer):
    def __init__(self, figure_id, new_figure, group, landmark_group):
        super(MayaviLandmarkViewer3d, self).__init__(figure_id, new_figure)
        self.group = group
        self.landmark_group = landmark_group

    def render(self, render_lines=True, line_colour='r', line_width=2,
               render_markers=True, marker_style='sphere', marker_size=None,
               marker_colour='r', marker_resolution=8, step=None, alpha=1.0,
               render_numbering=False, numbers_colour='k', numbers_size=None):
        # Regarding the labels colours, we may get passed either no colours (in
        # which case we generate random colours) or a single colour to colour
        # all the labels with
        # TODO: All marker and line options could be defined as lists...
        n_labels = self.landmark_group.n_labels
        line_colour = _check_colours_list(
            render_lines, line_colour, n_labels,
            'Must pass a list of line colours with length n_labels or a single '
            'line colour for all labels.')
        marker_colour = _check_colours_list(
            render_markers, marker_colour, n_labels,
            'Must pass a list of marker colours with length n_labels or a '
            'single marker face colour for all labels.')
        marker_size = _parse_marker_size(marker_size, self.landmark_group.points)
        numbers_size = _parse_marker_size(numbers_size,
                                          self.landmark_group.points)

        # get pointcloud of each label
        sub_pointclouds = self._build_sub_pointclouds()

        # for each pointcloud
        # disabling the rendering greatly speeds up this for loop
        self.figure.scene.disable_render = True
        for i, (label, pc) in enumerate(sub_pointclouds):
            # render pointcloud
            pc.view(figure_id=self.figure_id, new_figure=False,
                    render_lines=render_lines, line_colour=line_colour[i],
                    line_width=line_width, render_markers=render_markers,
                    marker_style=marker_style, marker_size=marker_size,
                    marker_colour=marker_colour[i],
                    marker_resolution=marker_resolution, step=step,
                    alpha=alpha, render_numbering=render_numbering,
                    numbers_colour=numbers_colour, numbers_size=numbers_size)
        self.figure.scene.disable_render = False

        return self

    def _build_sub_pointclouds(self):
        return [(label, self.landmark_group.get_label(label))
                for label in self.landmark_group.labels]
