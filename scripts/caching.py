import os
import numpy as np
import pickle
import logging
logger = logging.getLogger("CachedComputation")


def CachedComputation(fname, force=False):
    """ caches computation by pickling a dictionary with keys 'args', 'kwargs', 'results'. """
    def deep_equal(a, b):
        if type(a) != type(b):
            return False
        if not hasattr(a, "__len__"):
            return a == b
        if len(a) != len(b):
            return False
        if len(a) == 0:
            return True

        if isinstance(a, np.ndarray):
            try:
                return np.all(a == b)
            except Exception:
                return False

        if isinstance(a, list) or isinstance(a, tuple):
            for v_a, v_b in zip(a, b):
                if not deep_equal(v_a, v_b):
                    return False
            return True

        elif isinstance(a, dict):
            for k, v in a.items():
                if (k not in b) or not deep_equal(a[k], b[k]):
                    return False
            return True

        else:
            return a == b

        logger.error("shouldn't be here!")
        logger.error(f"type(a): {type(a)}")
        raise NotImplementedError("ugh")

    def decorator(orig_fun):
        def new_fun(*args, **kwargs):
            results = None
            if os.path.exists(fname):
                # load the results
                with open(fname, 'rb') as fin:
                    cache = pickle.load(fin)
                if deep_equal(cache['args'], args) and \
                   deep_equal(cache['kwargs'], kwargs):
                    results = cache['results']

            if (results is None) or force:
                results = orig_fun(*args, **kwargs)
            else:
                logger.info(f'Reusing cached outputs of {orig_fun.__name__}')

            # save results
            with open(fname, 'wb') as fout:
                pickle.dump({'args': args,
                             'kwargs': kwargs,
                             'results': results},
                            fout)
            return results

        return new_fun
    return decorator


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    def computation(a, b):
        print(f'computing {a} * {b}')
        return a * b

    def np_computation(a, b, c=3, method='usual'):
        print('--work done--')
        return np.matmul(a, b) + c

    print('non-cached:')
    print(computation(3, 4))

    ccomputation = CachedComputation("cache/debug.pkl")(computation)

    print('cached:')
    print(f"ccomputation(3, 4): {ccomputation(3, 4)}")

    print('\nnumpy with keyword arguments\n')
    cnp_computation = CachedComputation("cache/debug_np.pkl")(np_computation)
    x = np.array([[1, 2], [3, -2], [0.2, np.inf]])
    y = np.array([[2.5, 0.5], [1.2, 2]])
    z = np.array([[1, 0]])

    print(f"np_computation(x, y, z): {np_computation(x, y, z)}")
    print(f"cnp_computation(x, y, c=z, method='new'): {cnp_computation(x, y, c=z, method='new')}")

    from img_tools import sample_points_sequential
    import cv2
    img = cv2.imread('src_data/image/slovensko.jpg', 0)
    img = img.astype(np.float64) / 255

    sampling_fn = CachedComputation("cache/debug_sampling.pkl")(sample_points_sequential)
    print('sampling pts from image')
    pts = sampling_fn(img)
    print('done')
    print(f"pts.shape: {pts.shape}")
