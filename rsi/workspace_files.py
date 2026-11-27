"""Open model-owned workspace files without following links in the host process."""
import os
import stat
from contextlib import contextmanager


@contextmanager
def _parent(root, path, create=False):
    root = os.path.abspath(root)
    relative = os.path.relpath(os.path.abspath(path), root)
    if relative in ("..", ".") or relative.startswith(".." + os.sep):
        raise ValueError("path must name a file inside the workspace")
    *directories, name = relative.split(os.sep)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, flags)
    try:
        for directory in directories:
            if create:
                try:
                    os.mkdir(directory, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(directory, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, name
    finally:
        os.close(fd)


@contextmanager
def open_file(root, path, mode="r"):
    access = mode.removesuffix("b")
    with _parent(root, path, create=access != "r") as (parent, name):
        flags = {"r": os.O_RDONLY, "w": os.O_WRONLY | os.O_CREAT, "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND}[access]
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("workspace paths must be regular files with no additional links")
            if access == "w":
                os.ftruncate(fd, 0)
            text = {} if "b" in mode else {"encoding": "utf-8", "errors": "replace"}
            with os.fdopen(fd, mode, closefd=False, **text) as f:
                yield f
        finally:
            os.close(fd)


def read(root, path, limit=-1):
    with open_file(root, path) as f:
        return f.read(limit)


def write(root, path, content):
    with open_file(root, path, "w") as f:
        f.write(content)


def unlink(root, path):
    with _parent(root, path) as (parent, name):
        os.unlink(name, dir_fd=parent)
