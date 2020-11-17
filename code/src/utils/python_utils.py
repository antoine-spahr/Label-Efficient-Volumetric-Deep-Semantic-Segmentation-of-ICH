"""
author: Antoine Spahr

date : 27.10.2020

----------

TO DO :
- check if need to normalize output feature_maps in local contrastive task.
"""
import functools
import os
import json

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


class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed like attributes
        (as well as normally).
    """
    def __init__(self, *args, **kwargs):
        """
        Build a AttrDict from dict like this : AttrDict.from_nested_dicts(dict)
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_json_path(path):
        """ Construct nested AttrDicts from a json. """
        assert os.path.isfile(path), f'Path {path} does not exist.'
        with open(path, 'r') as fn:
            data = json.load(fn)
        return AttrDict.from_nested_dicts(data)

    @staticmethod
    def from_nested_dicts(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dicts(data[key])
                                for key in data})
