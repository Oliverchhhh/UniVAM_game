import os
import shutil

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        overwatch.info("Removing dirname: %s" % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError("Failed to delete %s because %s" % (dirname, e))

    if not os.path.exists(dirname):
        overwatch.info("Making dirname: %s" % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)


def ensure_directory(path):
    if path == "" or path == ".":
        return
    if path is not None and len(path) > 0:
        assert not os.path.isfile(path), "{} is a file".format(path)
        if not os.path.exists(path) and not os.path.islink(path):
            try:
                os.makedirs(path, exist_ok=True)
            except FileExistsError:
                # Ignore the exception since the directory already exists.
                pass
            except:  # noqa: E722
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
