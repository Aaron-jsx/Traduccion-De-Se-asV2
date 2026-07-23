__version__ = '1.7.1'

from .config import Config
from .visualization.color import color_val


def is_seq_of(val, type, seq_type=None):
    if seq_type is None:
        seq_type = (list, tuple)
    return isinstance(val, seq_type) and all(isinstance(v, type) for v in val)


class Timer:
    def __init__(self):
        import time
        self._start = time.time()

    def since_last_check(self):
        import time
        elapsed = time.time() - self._start
        self._start = time.time()
        return elapsed

    def since_start(self):
        import time
        return time.time() - self._start


def deprecated_api_warning(check_dict, cls_name=None):
    from .utils.misc import deprecated_api_warning as _deprecated
    return _deprecated(check_dict, cls_name)


def collect_env():
        return {}


def imread(img_or_path, flag='color'):
    from .image import imread as _imread
    return _imread(img_or_path, flag)


def imwrite(img, file_path, auto_mkdir=True):
    from .image import imwrite as _imwrite
    return _imwrite(img, file_path, auto_mkdir)


def imrescale(img, scale):
    from .image import imrescale as _imrescale
    return _imrescale(img, scale)


def bgr2rgb(img):
    from .image import bgr2rgb as _bgr2rgb
    return _bgr2rgb(img)


def rgb2bgr(img):
    from .image import rgb2bgr as _rgb2bgr
    return _rgb2bgr(img)


def imshow(img, win_name='', wait_time=0):
    from .visualization.image import imshow as _imshow
    return _imshow(img, win_name, wait_time)


def imshow_bboxes(img, bboxes, colors='green', top_k=-1, thickness=1, show=True, win_name='', wait_time=0):
    from .visualization.image import imshow_bboxes as _imshow_bboxes
    return _imshow_bboxes(img, bboxes, colors, top_k, thickness, show, win_name, wait_time)


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    import warnings as _w
    _w.warn(f'mmcv.{name} is not implemented (shim stub)')
    return _stub


class _stub:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return None
    def __getattr__(self, name): return self
