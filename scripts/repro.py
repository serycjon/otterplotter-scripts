# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import json
import contextlib
from datetime import datetime
import matplotlib.pyplot as plt
from svg import export_svg


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()


class ReproSaver():
    def __init__(self, storage='results/'):
        self.config = {}
        self.stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.storage = storage
        self.diff = git_diff()
        self.config['git_commit'] = git_commit()
        self.config['git_dirty'] = git_dirty_p()
        self.config['cmd_args'] = get_args()

    def seed(self):
        self.seed_python()
        self.seed_numpy()

    def seed_python(self):
        import random
        random.seed(self.stamp)
        self.config['rng_seed_python'] = self.stamp

    def seed_numpy(self):
        import random
        import numpy as np
        if 'rng_seed_python' not in self.config:
            self.seed_python()
        np_seed = random.randint(0, 2147483647)
        self.config['rng_seed_numpy'] = np_seed
        np.random.seed(np_seed)

    def add_plt_image(self, max_H=2160, max_W=3840):
        fig = plt.gcf()
        plt.tight_layout()
        size = fig.get_size_inches()
        dpi = scale_to_fit(size[0], size[1],
                           max_H, max_W)
        dpi = max(dpi, 600)
        out_dir = os.path.join(self.storage, 'vis')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, f'{self.stamp}.png'),
                    bbox_inches='tight', pad_inches=0,
                    dpi=dpi)
        self.save_info()

    def add_svg(self, drawing):
        out_dir = os.path.join(self.storage, 'svg')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        export_svg(drawing, os.path.join(out_dir, f'{self.stamp}.svg'))
        self.save_info()

    def add_info(self, key, value):
        self.config[key] = value

    def load_seeds(self, path):
        with open(path, 'r') as fin:
            config = json.loads(fin.read())
        rng_seed_python = config['rng_seed_python']
        rng_seed_numpy = config['rng_seed_numpy']
        import random
        random.seed(rng_seed_python)
        import numpy as np
        np.random.seed(rng_seed_numpy)

        self.config['rng_seed_python'] = rng_seed_python
        self.config['rng_seed_numpy'] = rng_seed_numpy

    def save_info(self):
        out_path = os.path.join(self.storage, 'meta', self.stamp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(os.path.join(out_path, 'config.json'), 'w') as fout:
            fout.write(json.dumps(self.config, indent=True, sort_keys=True))

        with open(os.path.join(out_path, 'git_diff'), 'wb') as fout:
            fout.write(self.diff)


@contextlib.contextmanager
def tmp_np_seed(seed):
    import numpy as np
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def run(args):
    path = None
    dirty = git_dirty_p(path)
    print('dirty: {}'.format(dirty))

    diff = git_diff(path)
    print('diff:\n' + diff.decode())

    in_args = get_args()
    print('in_args: {}'.format(in_args))

    commit = git_commit(path)
    print('commit: {}'.format(commit))
    return 0


def scale_to_fit(src_H, src_W, dst_H, dst_W):
    ''' Compute s, such that s*src_H, s*src_W fits in dst_H, dst_W '''
    s_H = dst_H / src_H
    s_W = dst_W / src_W

    return min(s_H, s_W)


def git_dirty_p(path=None):
    out = subprocess.check_output(["git", "status",
                                   # "-uno",  # ignore untracked
                                   "--porcelain"], cwd=path)
    return out.strip() != b""


def git_diff(path=None):
    out = subprocess.check_output(["git", "diff", "--no-color"], cwd=path)
    return out.strip()


def get_args():
    return sys.argv


def git_commit(path=None):
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path)
    return out.strip().decode()


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
