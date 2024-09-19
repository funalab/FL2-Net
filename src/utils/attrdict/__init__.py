"""
attrdict contains several mapping objects that allow access to their
keys as attributes.
"""
from src.utils.attrdict.mapping import AttrMap
from src.utils.attrdict.dictionary import AttrDict
from src.utils.attrdict.default import AttrDefault


__all__ = ['AttrMap', 'AttrDict', 'AttrDefault']
