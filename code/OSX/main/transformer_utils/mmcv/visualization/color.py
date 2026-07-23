import random


def color_val(val):
    if isinstance(val, str):
        from matplotlib import colors
        return colors.to_rgb(val)
    elif isinstance(val, (list, tuple)):
        return tuple(val)
    elif isinstance(val, (int, float)):
        return (val, val, val)
    return (0, 0, 0)
