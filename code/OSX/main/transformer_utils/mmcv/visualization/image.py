def imshow(img, win_name='', wait_time=0):
    import cv2
    cv2.namedWindow(win_name or 'image', cv2.WINDOW_NORMAL)
    cv2.imshow(win_name or 'image', img)
    cv2.waitKey(wait_time)


def imshow_bboxes(img, bboxes, colors='green', top_k=-1, thickness=1, show=True, win_name='', wait_time=0):
    import cv2
    img_copy = img.copy()
    bboxes_to_draw = bboxes[:top_k] if top_k > 0 else bboxes
    for bbox in bboxes_to_draw:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    if show:
        imshow(img_copy, win_name, wait_time)
