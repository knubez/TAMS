"""
Download examples to a temp directory and:

- compute sha256
- display dims and such
- show/check zlib settings and chunks for data variables
"""

import hashlib
from pathlib import Path
from tempfile import gettempdir
from textwrap import indent

TMP = Path(gettempdir()) / "tams-examples"

from tams.data import _EXAMPLE_FILES, _gdownload


def iter_blocks(p: Path, block_size=4096):
    with open(p, "rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            yield data


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    for block in iter_blocks(p):
        h.update(block)
    return h.hexdigest()


def check_file(p: Path) -> None:
    import xarray as xr

    nbytes = p.stat().st_size
    print(f"{p.name}: {nbytes / 1024**2:.1f} MiB ({nbytes / 1e6:.1f} MB)")
    print(f"  sha256: {sha256(p)}")

    ds = xr.open_dataset(p)
    print(indent(str(ds), "  "))

    w = max(len(str(vn)) for vn in ds.variables)
    print("  Encoding info:")
    for vn in ds.variables:
        da = ds[vn]
        enc = da.encoding
        print(
            f"    {vn:<{w}}  "
            f"zlib={enc.get('zlib')}, complevel={enc.get('complevel')}, "
            f"chunksizes={enc.get('chunksizes')}"
        )

    print("  Attributes:")
    for vn in ds.variables:
        s_attrs = ", ".join(f"{k}={v!r}" for k, v in ds[vn].attrs.items())
        if not s_attrs:
            s_attrs = "(empty)"
        print(f"    {vn:<{w}}  {s_attrs}")

    ds.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cached",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="if the file already exists in the temp directory, don't re-download it",
    )

    args = parser.parse_args()

    TMP.mkdir(exist_ok=True)
    for ef in _EXAMPLE_FILES:
        p = TMP / ef.fname
        if p.exists() and args.use_cached:
            print("Using cached")
        else:
            _gdownload(ef.file_id, p.as_posix(), quiet=False)
        check_file(p)
        print("\n")
