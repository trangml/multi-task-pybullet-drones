from collections import namedtuple

Bounds = namedtuple("Bounds", ["min", "max"])

POSITIVE_REWARD = 1
ZERO_REWARD = 0
NEGATIVE_REWARD = -1

def within_bounds(bounds: Bounds, field):
    return field > bounds.min and field < bounds.max