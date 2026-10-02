"""Validation for host directories used by memory transfer archives."""

import errno
import os
import platform
import stat
import sys


def validate_memory_transfer_dir(path: str, group_id: str) -> str | None:
    """Return an error message unless *path* is safe for memory transfer."""
    if platform.system() != "Linux":
        return "--memory-transfer-dir is supported only on Linux hosts"
    if not os.path.isabs(path):
        return f"--memory-transfer-dir must be an absolute path: {path}"
    if os.path.islink(path):
        return f"--memory-transfer-dir must not be a symbolic link: {path}"
    if not os.path.isdir(path):
        return f"--memory-transfer-dir must be a directory: {path}"
    if not group_id.isdecimal():
        return "--memory-transfer-gid must be a numeric private group ID"

    try:
        directory_stat = os.stat(path)
    except OSError:
        return f"Could not inspect --memory-transfer-dir: {path}"

    if directory_stat.st_uid not in (os.geteuid(), 0):
        return f"--memory-transfer-dir must be owned by the launching user or root: {path}"
    mode = stat.S_IMODE(directory_stat.st_mode)
    if not mode & stat.S_ISGID:
        return f"--memory-transfer-dir must enable setgid so archives inherit its private group: {path}"
    if _has_posix_acl(path):
        return f"--memory-transfer-dir must not grant access through ACL: {path}"
    if directory_stat.st_gid != int(group_id):
        return f"--memory-transfer-dir group must match --memory-transfer-gid: {path}"
    if mode & 0o007 or mode & 0o070 != 0o070:
        return (
            "--memory-transfer-dir must grant rwx to its private group and no access to other users: "
            f"{path}"
        )
    return None


def _has_posix_acl(path: str) -> bool:
    missing = {errno.ENODATA}
    unsupported = {errno.ENOTSUP, errno.EOPNOTSUPP}
    for name in ("system.posix_acl_access", "system.posix_acl_default"):
        try:
            os.getxattr(path, name, follow_symlinks=False)
        except OSError as exc:
            if exc.errno in missing:
                continue
            if exc.errno in unsupported:
                raise RuntimeError("Could not verify ACLs") from exc
            raise RuntimeError("Could not verify ACLs") from exc
        else:
            return True
    return False


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: memory_transfer.py <directory> <group-id>", file=sys.stderr)
        return 2
    try:
        error = validate_memory_transfer_dir(sys.argv[1], sys.argv[2])
    except RuntimeError as exc:
        print(f"{exc} for --memory-transfer-dir: {sys.argv[1]}", file=sys.stderr)
        return 1
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
