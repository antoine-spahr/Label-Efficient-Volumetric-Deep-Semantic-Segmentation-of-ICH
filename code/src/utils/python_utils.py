"""
author: Antoine Spahr

date : 27.10.2020

----------

TO DO :
- check if need to normalize output feature_maps in local contrastive task.
"""
import functools

def rgetattr(obj, attr, *args):
    """
    Extended getattr method to handle attribute access nested in parents classes : obj.attr1.attr2.attr3
    ----------
    INPUT
        |---- obj () the object to access attributes from.
        |---- attr (str) the attribute(s) name to access. Nested attributes are separated by a dot ('.')
    OUTPUT
        |---- obj_attr () the object attributes.
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
