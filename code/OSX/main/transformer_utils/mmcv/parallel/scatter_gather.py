def scatter_kwargs(*args, **kwargs):
    if args:
        return args, kwargs
    return (), kwargs
