"""
Model registry for lightweight 1D Poisson solvers
"""

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(spec, args=None):
    if args is not None:
        model = models[spec['name']](**args)
    else:
        model = models[spec['name']](**spec.get('args', {}))
    return model


# Import model classes to register them
from . import models


