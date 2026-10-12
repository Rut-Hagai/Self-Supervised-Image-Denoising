from .supervised import Supervised
from .noise2noise import Noise2Noise
from .noise2void import Noise2Void
from .neighbor2neighbor import Neighbor2Neighbor
from .blind2unblind import Blind2Unblind

REGISTRY = {s.name: s for s in
            [Supervised, Noise2Noise, Noise2Void, Neighbor2Neighbor, Blind2Unblind]}


def build(name, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
