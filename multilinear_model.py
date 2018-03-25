import numpy as np

# TODO: Should inherit from PCAModel (?)
class MultilinearModel(object):
    """Usage for a morphable model:
    
        shape_model = MultilinearModel(identity_model, expression_model)
        mm = TexturedMorphableModel(shape_model, texture_model, landmarks, tcoords, no_op, None)
    """
    def __init__(self, *models):
        self.models = models
        
    @property
    def n_active_components(self):
        return np.sum([mdl.n_active_components for mdl in self.models])

    @n_active_components.setter
    def n_active_components(self, values):
        try:
            for mdl, v in zip(self.models, values):
                mdl.n_active_components = value
        except TypeError:
            self.models[0].n_active_components = values

    def noise_variance(self):
        return self.models[0].noise_variance()

    def mean(self):
        v = np.sum([mdl.mean_vector for mdl in self.models], 0)

        return self.template_instance.from_vector(v)

    def instance(self, weights, normalized_weights=False):
        n_active = np.cumsum([0] + [mdl.n_active_components for mdl in self.models])
        slices = (slice(*s) for s in zip(n_active, n_active[1:]))
        
        v = np.sum([
                mdl.instance_vector(weights[start_end], normalized_weights) 
                for mdl, start_end in zip(self.models, slices)
            ], 0)
            
        
        return self.template_instance.from_vector(v)

    @property
    def components(self):
        return np.concatenate([mdl.components for mdl in self.models], 0)

    @property
    def template_instance(self):
        # TODO: Assert all models are of the same template instance.
        return self.models[0].template_instance
    
    @property
    def eigenvalues(self):
        return np.concatenate([mdl.eigenvalues for mdl in self.models], 0)
 
    def project(self, v):
        # Warning: Hacky way to project. Only projects onto the first
        # models components and sets the rest to zero.
        
        n_active_components = [mdl.n_active_components for mdl in self.models]
        e = np.zeros(sum(n_active_components[1:]))
        return np.hstack([self.models[0].project(v), e])