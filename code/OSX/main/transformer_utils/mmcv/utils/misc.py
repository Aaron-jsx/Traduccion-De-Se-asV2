import functools


def deprecated_api_warning(check_dict, cls_name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if cls_name:
                for old, new in check_dict.items():
                    if old in kwargs:
                        kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)
        return wrapper
    return decorator
