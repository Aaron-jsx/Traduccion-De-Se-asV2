"""
End-to-end pipeline: Video → OSX → SignAvatars .pkl → Rendered avatar.

Requires an NVIDIA GPU and mmcv-full/mmpose installed (see setup instructions below).

Setup (on a GPU machine):
    cd OSX
    pip install openmim
    mim install mmcv-full==1.7.1
    pip install -r requirements.txt
    cd main/transformer_utils && python setup.py install

    # Download pretrained model:
    # https://drive.google.com/drive/folders/1x7MZbB6eAlrq5PKC9MaeIm4GqkBpokow
    # Place at: OSX/pretrained_models/osx_l.pth.tar

    # Copy SMPL-X model files:
    cp -r ../common/utils/human_model_files common/utils/human_model_files

Usage:
    python pipeline/infer_video.py --video path/to/video.mp4 --render
"""

import common.compat  # noqa: F401 (must be first: Python 3.12 + NumPy 2 patches)
import os
import sys
import argparse
import numpy as np
import torch
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSX_DIR = os.path.join(ROOT, 'OSX')

sys.path.insert(0, OSX_DIR)
sys.path.insert(0, os.path.join(OSX_DIR, 'main'))
sys.path.insert(0, os.path.join(OSX_DIR, 'data'))
sys.path.insert(0, os.path.join(OSX_DIR, 'common'))

os.environ['DISPLAY'] = ':0.0'
os.environ["PYOPENGL_PLATFORM"] = "egl"


def init_osx():
    from config import cfg
    cfg.set_args('0')
    cfg.set_additional_args(
        encoder_setting='osx_l',
        decoder_setting='normal',
        pretrained_model_path=os.path.join(OSX_DIR, 'pretrained_models', 'osx_l.pth.tar')
    )

    from common.base import Demoer
    demoer = Demoer()
    demoer._make_model()
    assert os.path.exists(cfg.pretrained_model_path), \
        f'Download OSX model to {cfg.pretrained_model_path}'
    demoer.model.eval()
    return demoer


def detect_person(frame, detector):
    h, w = frame.shape[:2]
    with torch.no_grad():
        results = detector(frame)
    persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    boxes, confs = [], []
    for det in persons:
        x1, y1, x2, y2, conf, _ = det.tolist()
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confs.append(conf)
    if not boxes:
        return None, None
    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    return [boxes[i] for i in indices], [confs[i] for i in indices]


def run_osx_on_frame(frame, bbox, demoer, transform):
    from config import cfg
    from common.utils.preprocessing import process_bbox, generate_patch_image

    h, w = frame.shape[:2]
    bbox = process_bbox(bbox, w, h)
    if bbox is None:
        return None
    img_patch, _, bb2img_trans = generate_patch_image(frame, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    img_tensor = transform(img_patch.astype(np.float32)) / 255
    img_tensor = img_tensor.cuda()[None, :, :, :]

    with torch.no_grad():
        out = demoer.model({'img': img_tensor}, {}, {}, 'test')

    from pipeline.osx_to_signavatars import extract_osx_outputs
    frame_out = extract_osx_outputs(out)
    frame_out['bb2img_trans'] = bb2img_trans.astype(np.float32)
    return frame_out


def infer_video(video_path, output_pkl, max_frames=None, person_id=0):
    import torchvision.transforms as transforms
    from tqdm import tqdm

    print('Initializing OSX...')
    demoer = init_osx()

    print('Loading YOLOv5 detector...')
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    detector.conf = 0.5
    transform = transforms.ToTensor()

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Cannot open {video_path}'

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total = min(total, max_frames)

    all_frames = []
    pbar = tqdm(total=total, desc='OSX inference')

    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = detect_person(frame_rgb, detector)
        if boxes and person_id < len(boxes):
            out = run_osx_on_frame(frame_rgb, boxes[person_id], demoer, transform)
        else:
            out = None

        if out is not None:
            all_frames.append(out)
        elif all_frames:
            all_frames.append(all_frames[-1])
        else:
            from pipeline.osx_to_signavatars import extract_osx_outputs
            dummy = {'smplx_root_pose': torch.zeros(1, 3).cuda(),
                     'smplx_body_pose': torch.zeros(1, 63).cuda(),
                     'smplx_lhand_pose': torch.zeros(1, 45).cuda(),
                     'smplx_rhand_pose': torch.zeros(1, 45).cuda(),
                     'smplx_jaw_pose': torch.zeros(1, 3).cuda(),
                     'smplx_shape': torch.zeros(1, 10).cuda(),
                     'smplx_expr': torch.zeros(1, 10).cuda(),
                     'cam_trans': torch.zeros(1, 3).cuda(),
                     'smplx_joint_proj': torch.zeros(1, 137, 2).cuda()}
            dummy_out = extract_osx_outputs(dummy)
            dummy_out['bb2img_trans'] = np.eye(3, dtype=np.float32)[:2]
            all_frames.append(dummy_out)
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f'Processed {len(all_frames)} frames')

    from pipeline.osx_to_signavatars import build_pkl, save_pkl
    pkl_data = build_pkl(all_frames, width, height, use_consistent_shape=True)
    save_pkl(pkl_data, output_pkl)
    return output_pkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video → OSX → SignAvatars avatar')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', default=None, help='Output .pkl path')
    parser.add_argument('--max_frames', type=int, default=None, help='Max frames')
    parser.add_argument('--person_id', type=int, default=0, help='Which detected person to track')
    parser.add_argument('--render', action='store_true', help='Render result with vis.py')
    parser.add_argument('--overlay', action='store_true', help='Overlay on video')
    parser.add_argument('--video_path', default=None, help='Video for overlay')
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out = args.output or os.path.join(ROOT, 'render_results', f'{video_name}.pkl')

    infer_video(args.video, out, args.max_frames, args.person_id)

    if args.render:
        os.chdir(ROOT)
        sys.argv = ['vis.py', '--pkl_file_path', out]
        if args.overlay:
            vid = args.video_path or args.video
            sys.argv += ['--overlay', '--video_path', vid]
        exec(open(os.path.join(ROOT, 'vis.py')).read())
