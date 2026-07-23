import numpy as np
import pickle
import os


SMPLX_DIM = 182
UNSMOOTH_DIM = 169
NUM_OSX_JOINTS = 137


def stack_frames(frames, key, fill_value=None):
    values = [f.get(key) for f in frames]
    if fill_value is not None and any(v is None for v in values):
        values = [v if v is not None else fill_value for v in values]
    return np.stack(values, axis=0).astype(np.float32)


def build_pkl(frame_outputs, width, height, focal_length=None, princpt=None,
              use_consistent_shape=True):
    N = len(frame_outputs)
    if N == 0:
        raise ValueError('frame_outputs is empty')

    root = stack_frames(frame_outputs, 'root_pose', np.zeros(3))
    body = stack_frames(frame_outputs, 'body_pose', np.zeros(63))
    lhand = stack_frames(frame_outputs, 'lhand_pose', np.zeros(45))
    rhand = stack_frames(frame_outputs, 'rhand_pose', np.zeros(45))
    jaw = stack_frames(frame_outputs, 'jaw_pose', np.zeros(3))
    shape = stack_frames(frame_outputs, 'shape', np.zeros(10))
    expr = stack_frames(frame_outputs, 'expr', np.zeros(10))
    cam = stack_frames(frame_outputs, 'cam_trans', np.zeros(3))

    if use_consistent_shape and N > 1:
        shape_mean = shape.mean(axis=0, keepdims=True)
        shape = np.tile(shape_mean, (N, 1))

    smplx_182 = np.concatenate([root, body, lhand, rhand, jaw, shape, expr, cam], axis=1)
    unsmooth = np.concatenate([root, body, lhand, rhand, shape, cam], axis=1)

    joint_proj = stack_frames(frame_outputs, 'joint_proj', np.zeros((NUM_OSX_JOINTS, 2)))
    joint_3d = np.concatenate([joint_proj, np.ones((N, NUM_OSX_JOINTS, 1), dtype=np.float32)], axis=2)

    bb2img = stack_frames(frame_outputs, 'bb2img_trans', np.eye(3)[:2])

    if focal_length is None:
        focal_length = np.array([[5000, 5000]], dtype=np.float32)
    if princpt is None:
        princpt = np.array([[192 / 2, 256 / 2]], dtype=np.float32)

    data = {
        'width': np.array([width]),
        'height': np.array([height]),
        'focal': np.tile(focal_length, (N, 1)),
        'princpt': np.tile(princpt, (N, 1)),
        '2d': joint_3d.copy(),
        'pred2d': joint_3d.copy(),
        'total_valid_index': np.ones(N, dtype=np.int64),
        'left_valid': np.ones(N, dtype=np.int64),
        'right_valid': np.ones(N, dtype=np.int64),
        'bb2img_trans': bb2img,
        'smplx': smplx_182.astype(np.float32),
        'unsmooth_smplx': unsmooth.astype(np.float32),
    }
    return data


def save_pkl(data, output_path):
    out_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    N = data['smplx'].shape[0]
    S = data['smplx'].shape
    print(f'Saved: {output_path} ({N} frames, smplx shape: {S})')


def extract_osx_outputs(out_dict):
    return {
        'root_pose': out_dict['smplx_root_pose'].detach().cpu().numpy()[0],
        'body_pose': out_dict['smplx_body_pose'].detach().cpu().numpy()[0],
        'lhand_pose': out_dict['smplx_lhand_pose'].detach().cpu().numpy()[0],
        'rhand_pose': out_dict['smplx_rhand_pose'].detach().cpu().numpy()[0],
        'jaw_pose': out_dict['smplx_jaw_pose'].detach().cpu().numpy()[0],
        'shape': out_dict['smplx_shape'].detach().cpu().numpy()[0],
        'expr': out_dict['smplx_expr'].detach().cpu().numpy()[0],
        'cam_trans': out_dict['cam_trans'].detach().cpu().numpy()[0],
        'joint_proj': out_dict['smplx_joint_proj'].detach().cpu().numpy()[0],
        'bb2img_trans': None,
    }


def parse_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    smplx_params = data['smplx']
    frames = []
    for i in range(smplx_params.shape[0]):
        p = smplx_params[i]
        frames.append({
            'root_pose': p[0:3],
            'body_pose': p[3:66],
            'lhand_pose': p[66:111],
            'rhand_pose': p[111:156],
            'jaw_pose': p[156:159],
            'shape': p[159:169],
            'expr': p[169:179],
            'cam_trans': p[179:182],
        })
    w = data.get('width', [640])[0]
    h = data.get('height', [480])[0]
    return frames, w, h
