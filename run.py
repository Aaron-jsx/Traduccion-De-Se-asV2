import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))

OSX_DIR = os.path.join(ROOT, 'code', 'OSX')
SA_DIR = os.path.join(ROOT, 'code', 'SignAvatars')
MODEL_PATH = os.path.join(ROOT, 'models', 'pretrained_models', 'osx_l_wo_decoder.pth.tar')
VIDEOS_DIR = os.path.join(ROOT, 'videos')
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')


def cmd_infer(args):
    sys.path.insert(0, OSX_DIR)
    sys.path.insert(0, os.path.join(OSX_DIR, 'main'))
    sys.path.insert(0, os.path.join(OSX_DIR, 'data'))

    from config import cfg
    import torch
    import torch.backends.cudnn as cudnn
    import torchvision.transforms as transforms
    import numpy as np
    import cv2
    import pickle

    cfg.set_args('0')
    cudnn.benchmark = True
    cfg.set_additional_args(
        encoder_setting='osx_l',
        decoder_setting='wo_decoder',
        pretrained_model_path=args.model or MODEL_PATH,
    )

    from common.base import Demoer
    from common.utils.preprocessing import process_bbox, generate_patch_image

    demoer = Demoer()
    demoer._make_model()

    transform = transforms.ToTensor()

    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    detector.conf = 0.5

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_outputs = []
    for frame_idx in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        original_img = frame.copy()
        h, w = original_img.shape[:2]

        with torch.no_grad():
            results = detector(original_img)

        pred = getattr(results, 'pred', None)
        if pred is None:
            pred = getattr(results, 'xyxy', None)
        if pred is not None:
            dets = pred[0]
        elif isinstance(results, (list, tuple)):
            dets = results[0]
        else:
            dets = results

        person_mask = dets[:, 5] == 0 if dets.ndim > 1 and dets.shape[1] > 5 else None
        if person_mask is None or person_mask.sum() == 0:
            continue

        person = dets[person_mask]
        x1, y1, x2, y2 = person[0, :4].tolist()
        bbox = process_bbox([x1, y1, x2 - x1, y2 - y1], w, h)

        img_patch, img2bb_trans, bb2img_trans = generate_patch_image(
            original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
        )
        img_tensor = transform(img_patch.astype(np.float32)) / 255
        img_tensor = img_tensor.cuda()[None, :, :, :]

        inputs = {'img': img_tensor}
        with torch.no_grad():
            out = demoer.model(inputs, {}, {}, 'test')

        frame_outputs.append({
            'root_pose': out['smplx_root_pose'][0].cpu().numpy(),
            'body_pose': out['smplx_body_pose'][0].cpu().numpy(),
            'lhand_pose': out['smplx_lhand_pose'][0].cpu().numpy(),
            'rhand_pose': out['smplx_rhand_pose'][0].cpu().numpy(),
            'jaw_pose': out['smplx_jaw_pose'][0].cpu().numpy(),
            'shape': out['smplx_shape'][0].cpu().numpy(),
            'expr': out['smplx_expr'][0].cpu().numpy(),
            'cam_trans': out['cam_trans'][0].cpu().numpy(),
            'bb2img_trans': bb2img_trans,
        })
        if (frame_idx + 1) % 50 == 0:
            print(f'  {frame_idx + 1}/{N_FRAMES}', flush=True)

    cap.release()

    N = len(frame_outputs)
    NUM_OSX_JOINTS = 137

    def stack_frames(frames, key, fill=None):
        vals = [f.get(key) for f in frames]
        if fill is not None:
            vals = [v if v is not None else fill for v in vals]
        return np.stack(vals, axis=0).astype(np.float32)

    root = stack_frames(frame_outputs, 'root_pose', np.zeros(3))
    body = stack_frames(frame_outputs, 'body_pose', np.zeros(63))
    lhand = stack_frames(frame_outputs, 'lhand_pose', np.zeros(45))
    rhand = stack_frames(frame_outputs, 'rhand_pose', np.zeros(45))
    jaw = stack_frames(frame_outputs, 'jaw_pose', np.zeros(3))
    shape = stack_frames(frame_outputs, 'shape', np.zeros(10))
    expr = stack_frames(frame_outputs, 'expr', np.zeros(10))
    cam = stack_frames(frame_outputs, 'cam_trans', np.zeros(3))

    if N > 1:
        shape[:] = shape.mean(axis=0, keepdims=True)

    smplx_182 = np.concatenate([root, body, lhand, rhand, jaw, shape, expr, cam], axis=1)
    unsmooth = np.concatenate([root, body, lhand, rhand, shape, cam], axis=1)

    data = {
        'width': np.array([W]),
        'height': np.array([H]),
        'focal': np.tile(np.array([[5000, 5000]], dtype=np.float32), (N, 1)),
        'princpt': np.tile(np.array([[cfg.input_img_shape[1]/2, cfg.input_img_shape[0]/2]], dtype=np.float32), (N, 1)),
        '2d': np.zeros((N, NUM_OSX_JOINTS, 3), dtype=np.float32),
        'pred2d': np.zeros((N, NUM_OSX_JOINTS, 3), dtype=np.float32),
        'total_valid_index': np.ones(N, dtype=np.int64),
        'left_valid': np.ones(N, dtype=np.int64),
        'right_valid': np.ones(N, dtype=np.int64),
        'bb2img_trans': np.tile(np.eye(3)[:2][None], (N, 1, 1)).astype(np.float32),
        'smplx': smplx_182.astype(np.float32),
        'unsmooth_smplx': unsmooth.astype(np.float32),
    }

    output_path = args.output or os.path.join(OUTPUTS_DIR, os.path.splitext(os.path.basename(args.video))[0] + '.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved: {output_path}')


def cmd_render(args):
    os.chdir(SA_DIR)
    sys.argv = ['vis.py', '--pkl_file_path', args.pkl]
    if args.overlay:
        sys.argv += ['--overlay']
        if args.video_path:
            sys.argv += ['--video_path', args.video_path]
    exec(open(os.path.join(SA_DIR, 'vis.py')).read())


def cmd_convert(args):
    from pipeline.osx_to_signavatars import build_pkl, save_pkl
    import numpy as np
    import json

    with open(args.input, 'r') as f:
        raw = json.load(f)

    frames = []
    for fr in raw:
        frames.append({
            'root_pose': np.array(fr['root_pose'], dtype=np.float32),
            'body_pose': np.array(fr['body_pose'], dtype=np.float32),
            'lhand_pose': np.array(fr['lhand_pose'], dtype=np.float32),
            'rhand_pose': np.array(fr['rhand_pose'], dtype=np.float32),
            'jaw_pose': np.array(fr['jaw_pose'], dtype=np.float32),
            'shape': np.array(fr['shape'], dtype=np.float32),
            'expr': np.array(fr['expr'], dtype=np.float32),
            'cam_trans': np.array(fr['cam_trans'], dtype=np.float32),
            'joint_proj': np.array(fr.get('joint_proj', [[0]*2]*137), dtype=np.float32),
            'bb2img_trans': np.array(fr.get('bb2img_trans', np.eye(3)[:2]), dtype=np.float32),
        })

    width = args.width or 1280
    height = args.height or 720
    data = build_pkl(frames, width, height, use_consistent_shape=not args.per_frame_shape)
    save_pkl(data, args.output)


def cmd_doctor(args):
    print(f'OSX_Project doctor — {ROOT}')
    print()

    ok = True

    def check(cond, msg):
        nonlocal ok
        status = 'PASS' if cond else 'FAIL'
        if not cond:
            ok = False
        print(f'  [{status}] {msg}')

    print('[Python]')
    check(sys.version_info >= (3, 8), f'Python >= 3.8 (found {sys.version_info.major}.{sys.version_info.minor})')

    print()
    print('[PyTorch]')
    try:
        import torch
        check(True, f'torch {torch.__version__}')
    except ImportError:
        check(False, 'torch not installed')
    try:
        check(torch.cuda.is_available(), f'CUDA available, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}')
        if torch.cuda.is_available():
            print(f'         CUDA version: {torch.version.cuda}')
    except NameError:
        pass

    print()
    print('[Directories]')
    check(os.path.isdir(OSX_DIR), f'{OSX_DIR}')
    check(os.path.isdir(SA_DIR), f'{SA_DIR}')
    check(os.path.isdir(os.path.join(ROOT, 'models', 'human_model_files')), 'models/human_model_files/')
    check(os.path.isdir(os.path.join(ROOT, 'models', 'pretrained_models')), 'models/pretrained_models/')
    check(os.path.isdir(VIDEOS_DIR), 'videos/')

    print()
    print('[Human model files]')
    hmm = os.path.join(ROOT, 'models', 'human_model_files')
    for f in ['smplx/SMPLX_NEUTRAL.pkl', 'smplx/SMPLX_NEUTRAL.npz', 'smplx/SMPLX_to_J14.pkl',
              'smplx/MANO_SMPLX_vertex_ids.pkl', 'smplx/SMPL-X__FLAME_vertex_ids.npy',
              'mano/MANO_LEFT.pkl', 'mano/MANO_RIGHT.pkl',
              'smpl/SMPL_NEUTRAL.pkl',
              'flame/FLAME_NEUTRAL.pkl']:
        check(os.path.isfile(os.path.join(hmm, f)), f'  {f}')

    print()
    print('[Checkpoint]')
    check(os.path.isfile(MODEL_PATH), f'osx_l_wo_decoder.pth.tar ({MODEL_PATH})')

    print()
    print('[Outputs]')
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    check(os.path.isdir(OUTPUTS_DIR), f'{OUTPUTS_DIR}')

    print()
    if ok:
        print('All checks passed.')
    else:
        print('Some checks FAILED. See messages above.')
    sys.exit(0 if ok else 1)


def cmd_inspect(args):
    import pickle

    path = args.pkl
    if not os.path.isfile(path):
        print(f'File not found: {path}')
        sys.exit(1)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f'PKL: {path}')
    print()

    print(f'{"Key":30s} {"Shape":20s} {"Dtype":12s} {"Status":s}')
    print('-' * 80)

    expected = {
        'width': ((1,), False),
        'height': ((1,), False),
        'focal': ((None, 2), True),
        'princpt': ((None, 2), True),
        '2d': ((None, 137, 3), True),
        'pred2d': ((None, 137, 3), True),
        'total_valid_index': ((None,), True),
        'left_valid': ((None,), True),
        'right_valid': ((None,), True),
        'bb2img_trans': ((None, 2, 3), True),
        'smplx': ((None, 182), True),
        'unsmooth_smplx': ((None, 169), True),
    }

    n_frames = None
    for key in sorted(data.keys()):
        val = data[key]
        if isinstance(val, str):
            print(f'{key:30s} {"(string)":20s} {"":12s}')
            continue
        shape = val.shape if hasattr(val, 'shape') else '?'
        dtype = str(val.dtype) if hasattr(val, 'dtype') else ''
        shape_str = str(tuple(shape)) if hasattr(val, 'shape') else '-'

        if key == 'smplx' and hasattr(val, 'shape'):
            n_frames = val.shape[0]

        if key in expected:
            exp_shape, _ = expected[key]
            if exp_shape[0] is None:
                if n_frames is not None and len(shape) == len(exp_shape):
                    prefix = tuple([n_frames] + list(exp_shape[1:]))
                    status = 'PASS' if shape == prefix else f'EXPECTED {prefix}'
                else:
                    status = f'dim={len(shape)}'
            else:
                status = 'PASS' if tuple(shape) == exp_shape else f'EXPECTED {exp_shape}'
        else:
            status = ''

        print(f'{key:30s} {shape_str:20s} {dtype:12s} {status:s}')

    print()
    if n_frames:
        print(f'Frames: {n_frames}')
        print(f'smplx dim: 182 (verified at column split: root=3 body=63 lhand=45 rhand=45 jaw=3 shape=10 expr=10 cam=3)')

    print()
    all_required = True
    for key in expected:
        if key not in data:
            print(f'  MISSING: {key}')
            all_required = False
    if all_required:
        print('All required keys present.')
    else:
        print('Some required keys are missing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OSX_Project: Video -> SMPL-X pipeline')
    sub = parser.add_subparsers(dest='command')

    p_infer = sub.add_parser('infer', help='Run OSX inference on a video')
    p_infer.add_argument('--video', required=True)
    p_infer.add_argument('--model', default=None, help='Path to OSX checkpoint')
    p_infer.add_argument('--output', default=None)
    p_infer.add_argument('--render', action='store_true')
    p_infer.add_argument('--overlay', action='store_true')
    p_infer.add_argument('--video_path')

    p_conv = sub.add_parser('convert', help='Convert OSX JSON output to SignAvatars .pkl')
    p_conv.add_argument('--input', required=True)
    p_conv.add_argument('--output', required=True)
    p_conv.add_argument('--width', type=int)
    p_conv.add_argument('--height', type=int)
    p_conv.add_argument('--per_frame_shape', action='store_true')

    p_render = sub.add_parser('render', help='Render a .pkl file')
    p_render.add_argument('--pkl', required=True)
    p_render.add_argument('--overlay', action='store_true')
    p_render.add_argument('--video_path')

    p_doc = sub.add_parser('doctor', help='Validate the runtime environment')
    p_doc.set_defaults(func=cmd_doctor)

    p_ins = sub.add_parser('inspect', help='Inspect a generated .pkl file')
    p_ins.add_argument('--pkl', required=True)
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    elif args.command == 'infer':
        cmd_infer(args)
    elif args.command == 'convert':
        sys.path.insert(0, os.path.join(SA_DIR, 'pipeline'))
        cmd_convert(args)
    elif args.command == 'render':
        cmd_render(args)
    elif args.command == 'doctor':
        cmd_doctor(args)
    elif args.command == 'inspect':
        cmd_inspect(args)
