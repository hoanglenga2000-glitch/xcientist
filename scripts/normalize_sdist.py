from __future__ import annotations

import argparse
import copy
import gzip
import io
import os
import tarfile
from pathlib import Path, PurePosixPath


def normalize_sdist(path: Path, epoch: int) -> None:
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be non-negative")

    tar_buffer = io.BytesIO()
    with tarfile.open(path, mode="r:gz") as source, tarfile.open(
        fileobj=tar_buffer,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as target:
        members = sorted(source.getmembers(), key=lambda member: member.name)
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("sdist contains duplicate member names")
        for original in members:
            member_path = PurePosixPath(original.name)
            if member_path.is_absolute() or ".." in member_path.parts or "\\" in original.name:
                raise ValueError(f"unsafe sdist member path: {original.name!r}")
            if not (original.isfile() or original.isdir()):
                raise ValueError(f"unsupported sdist member type: {original.name!r}")
            member = copy.copy(original)
            member.mtime = epoch
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.pax_headers = {
                key: value
                for key, value in member.pax_headers.items()
                if key not in {"atime", "ctime", "mtime"}
            }
            fileobj = source.extractfile(original) if original.isfile() else None
            try:
                target.addfile(member, fileobj=fileobj)
            finally:
                if fileobj is not None:
                    fileobj.close()

    temporary = path.with_name(f".{path.name}.normalized.tmp")
    with temporary.open("wb") as output, gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=output,
        mtime=epoch,
    ) as compressed:
        compressed.write(tar_buffer.getvalue())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a Python sdist for reproducible release hashing.")
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    normalize_sdist(args.sdist, args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
