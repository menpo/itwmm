import numpy as np

from menpo.shape import PointCloud
from menpo.transform.base import Transformable


class ShapeModel(Transformable):
    r"""
    Wrapper around a :map:`PCAModel` that makes it transformable and
    maskable in both dimensions and points.
    """
    def __init__(self, model):
        self.model = model

    @property
    def n_dims(self):
        return self.model.template_instance.n_dims

    @property
    def n_components(self):
        return self.model.n_components

    @property
    def _c(self):
        return self.model._components.reshape(
            [self.n_components, -1, self.n_dims])

    def mask_points(self, mask):
        new_model = self.model.copy()
        new_model._components = self._c[:, mask].reshape(
            [self.n_components, -1])
        mean = self.model.mean().points[mask]
        new_model._mean = mean.ravel()
        new_model.template_instance = PointCloud(mean)
        return ShapeModel(new_model)

    def mask_dims(self, mask):
        new_model = self.model.copy()
        new_model._components = self._c[..., mask].reshape(
            [self.n_components, -1])
        mean = self.model.mean().points[:, mask]
        new_model._mean = mean.ravel()
        new_model.template_instance = PointCloud(mean)
        return ShapeModel(new_model)

    def _transform(self, transform):
        new_model = self.model.copy()
        new_model._components = np.concatenate(
            [transform(c)[None] for c in self._c]).reshape(
            [self.n_components, -1])
        new_model._mean = transform(self.model.mean().points).ravel()
        return ShapeModel(new_model)

    def project(self, instance, n_components=None):
        if n_components is not None:
            prev_n_components = self.model.n_active_components
            self.model.n_active_components = n_components
        weights = np.linalg.lstsq(self.model.components.T,
                                  instance.as_vector() - self.model.mean_vector)[
            0]
        if n_components is not None:
            self.model.n_active_components = prev_n_components
        return weights
