import os


def imread(img_or_path, flag='color'):
    import cv2
    if isinstance(img_or_path, str):
        return cv2.imread(img_or_path, cv2.IMREAD_COLOR if flag == 'color' else cv2.IMREAD_GRAYSCALE)
    return img_or_path


def imwrite(img, file_path, auto_mkdir=True):
    import cv2
    if auto_mkdir:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return cv2.imwrite(file_path, img)


def imrescale(img, scale):
    import cv2
    h, w = img.shape[:2]
    if isinstance(scale, (int, float)):
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_w, new_h = scale
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def bgr2rgb(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2bgr(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
