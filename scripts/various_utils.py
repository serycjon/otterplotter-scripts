import traceback
import ipdb
import sys

try:
    from OtterPlotter import ConnectionError
except ImportError:
    class ConnectionError(Exception):
        pass


def with_debugger(orig_fn):
    def new_fn(*args, **kwargs):
        try:
            return orig_fn(*args, **kwargs)
        except ConnectionError:
            print("Cannot connect to OtterPlotter!", file=sys.stderr)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            ipdb.post_mortem()

    return new_fn
