import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def cmd(args):
    if args.command == 'convert':
        from pipeline.osx_to_signavatars import build_pkl, save_pkl, parse_pkl
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

    elif args.command == 'render':
        os.chdir(ROOT)
        sys.argv = ['vis.py', '--pkl_file_path', args.pkl]
        if args.overlay:
            sys.argv += ['--overlay']
            if args.video_path:
                sys.argv += ['--video_path', args.video_path]
        exec(open(os.path.join(ROOT, 'vis.py')).read())

    elif args.command == 'infer':
        sys.path.insert(0, os.path.join(ROOT, 'OSX'))
        sys.path.insert(0, os.path.join(ROOT, 'OSX', 'main'))
        from pipeline.infer_video import infer_video
        out = args.output or os.path.join(ROOT, 'render_results', os.path.splitext(os.path.basename(args.video))[0] + '.pkl')
        infer_video(args.video, out, args.max_frames, args.person_id)

        if args.render:
            os.chdir(ROOT)
            sys.argv = ['vis.py', '--pkl_file_path', out]
            if args.overlay:
                sys.argv += ['--overlay']
                vid = args.video_path or args.video
                sys.argv += ['--video_path', vid]
            exec(open(os.path.join(ROOT, 'vis.py')).read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignAvatars Pipeline: video <-> avatar')
    sub = parser.add_subparsers(dest='command')

    p_infer = sub.add_parser('infer', help='Video -> OSX -> .pkl (requires GPU)')
    p_infer.add_argument('--video', required=True)
    p_infer.add_argument('--output')
    p_infer.add_argument('--max_frames', type=int)
    p_infer.add_argument('--person_id', type=int, default=0)
    p_infer.add_argument('--render', action='store_true')
    p_infer.add_argument('--overlay', action='store_true')
    p_infer.add_argument('--video_path')

    p_conv = sub.add_parser('convert', help='JSON -> .pkl (from saved OSX outputs)')
    p_conv.add_argument('--input', required=True, help='JSON file with per-frame OSX outputs')
    p_conv.add_argument('--output', required=True)
    p_conv.add_argument('--width', type=int)
    p_conv.add_argument('--height', type=int)
    p_conv.add_argument('--per_frame_shape', action='store_true')

    p_render = sub.add_parser('render', help='.pkl -> avatar video')
    p_render.add_argument('--pkl', required=True)
    p_render.add_argument('--overlay', action='store_true')
    p_render.add_argument('--video_path')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        cmd(args)
