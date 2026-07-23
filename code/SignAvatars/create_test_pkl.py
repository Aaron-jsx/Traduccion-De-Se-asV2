import numpy as np
import pickle
import os

num_frames = 60
width, height = 1280, 720

focal = np.tile(np.array([[14921.82254791, 14921.82254791]]), (num_frames, 1))
princpt = np.tile(np.array([[620.60418701, 413.40108109]]), (num_frames, 1))
total_valid_index = np.ones(num_frames, dtype=np.int64)
left_valid = np.ones(num_frames, dtype=np.int64)
right_valid = np.ones(num_frames, dtype=np.int64)

bb2img_trans = np.tile(np.eye(3)[:2], (num_frames, 1, 1))

root_pose = np.zeros((num_frames, 3))
body_pose = np.zeros((num_frames, 63))
lhand_pose = np.zeros((num_frames, 45))
rhand_pose = np.zeros((num_frames, 45))
jaw_pose = np.zeros((num_frames, 3))
shape = np.zeros((num_frames, 10))
expression = np.zeros((num_frames, 10))
cam_trans = np.zeros((num_frames, 3))

body_pose[:, 0] = 0.1
body_pose[:, 3] = 0.05

t = np.arange(num_frames) / num_frames
lhand_pose[:, 0] = 0.5 * np.sin(t * 2 * np.pi)
rhand_pose[:, 0] = 0.3 * np.cos(t * 2 * np.pi)

smplx_params = np.concatenate([root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expression, cam_trans], axis=1)
unsmooth_smplx = np.concatenate([root_pose, body_pose, lhand_pose, rhand_pose, shape, cam_trans], axis=1)

data = {
    'width': np.array([width]),
    'height': np.array([height]),
    'focal': focal,
    'princpt': princpt,
    '2d': np.zeros((num_frames, 106, 3)),
    'pred2d': np.zeros((num_frames, 106, 3)),
    'total_valid_index': total_valid_index,
    'left_valid': left_valid,
    'right_valid': right_valid,
    'bb2img_trans': bb2img_trans,
    'smplx': smplx_params.astype(np.float32),
    'unsmooth_smplx': unsmooth_smplx.astype(np.float32),
}

os.makedirs('test_data', exist_ok=True)
with open('test_data/test_sample.pkl', 'wb') as f:
    pickle.dump(data, f)
print(f"Created test_data/test_sample.pkl with {num_frames} frames, smplx shape: {smplx_params.shape}")
