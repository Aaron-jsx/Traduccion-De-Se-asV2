import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def check(ok, msg):
    status = 'PASS' if ok else 'FAIL'
    print(f'  [{status}] {msg}')
    return ok


def main():
    all_pass = True
    print('OSX_Project — Environment Validation')
    print()

    # Python version
    print('[1] Python')
    all_pass &= check(sys.version_info >= (3, 8), f'Python >= 3.8 (found {sys.version_info.major}.{sys.version_info.minor})')

    print()
    print('[2] PyTorch')
    try:
        import torch
        all_pass &= check(True, f'PyTorch {torch.__version__} installed')
        all_pass &= check(torch.cuda.is_available(), 'CUDA available')
        if torch.cuda.is_available():
            print(f'         CUDA version: {torch.version.cuda}')
            print(f'         Device count: {torch.cuda.device_count()}')
            print(f'         Device name:  {torch.cuda.get_device_name(0)}')
    except ImportError:
        all_pass &= check(False, 'PyTorch not installed (run: pip install torch torchvision)')

    print()
    print('[3] Project directories')
    checks = [
        ('code/OSX', os.path.isdir(os.path.join(ROOT, 'code', 'OSX'))),
        ('code/SignAvatars', os.path.isdir(os.path.join(ROOT, 'code', 'SignAvatars'))),
        ('models/human_model_files', os.path.isdir(os.path.join(ROOT, 'models', 'human_model_files'))),
        ('models/pretrained_models', os.path.isdir(os.path.join(ROOT, 'models', 'pretrained_models'))),
        ('videos', os.path.isdir(os.path.join(ROOT, 'videos'))),
        ('outputs', os.path.isdir(os.path.join(ROOT, 'outputs'))),
    ]
    for name, ok in checks:
        all_pass &= check(ok, f'{name}/')

    print()
    print('[4] Model files')
    expected = [
        ('SMPLX_NEUTRAL.pkl', os.path.join(ROOT, 'models', 'human_model_files', 'smplx', 'SMPLX_NEUTRAL.pkl')),
        ('SMPLX_NEUTRAL.npz', os.path.join(ROOT, 'models', 'human_model_files', 'smplx', 'SMPLX_NEUTRAL.npz')),
        ('MANO_LEFT.pkl', os.path.join(ROOT, 'models', 'human_model_files', 'mano', 'MANO_LEFT.pkl')),
        ('MANO_RIGHT.pkl', os.path.join(ROOT, 'models', 'human_model_files', 'mano', 'MANO_RIGHT.pkl')),
        ('SMPL_NEUTRAL.pkl', os.path.join(ROOT, 'models', 'human_model_files', 'smpl', 'SMPL_NEUTRAL.pkl')),
        ('SMPLX_to_J14.pkl', os.path.join(ROOT, 'models', 'human_model_files', 'smplx', 'SMPLX_to_J14.pkl')),
        ('MANO_SMPLX_vertex_ids.pkl', os.path.join(ROOT, 'models', 'human_model_files', 'smplx', 'MANO_SMPLX_vertex_ids.pkl')),
        ('SMPL-X__FLAME_vertex_ids.npy', os.path.join(ROOT, 'models', 'human_model_files', 'smplx', 'SMPL-X__FLAME_vertex_ids.npy')),
        ('FLAME_NEUTRAL.pkl', os.path.join(ROOT, 'models', 'human_model_files', 'flame', 'FLAME_NEUTRAL.pkl')),
    ]
    for name, path in expected:
        all_pass &= check(os.path.isfile(path), f'   {name}')

    print()
    print('[5] Checkpoint')
    ckpt = os.path.join(ROOT, 'models', 'pretrained_models', 'osx_l_wo_decoder.pth.tar')
    all_pass &= check(os.path.isfile(ckpt), 'osx_l_wo_decoder.pth.tar')

    print()
    if all_pass:
        print('All checks PASSED — environment is ready.')
        return 0
    else:
        print('Some checks FAILED. Fix the issues above before running the pipeline.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
