"""
Dataset registry for lightweight 1D Poisson solvers
"""

datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(spec, args=None):
    if args is not None:
        dataset = datasets[spec['name']](**args)
    else:
        dataset = datasets[spec['name']](**spec['args'])
    return dataset


# Import dataset classes to register them
from . import poisson1d_synthetic


