#!/usr/bin/env python3
"""Install the pinned sqlite-vec loadable extension for this platform."""

from __future__ import annotations

import hashlib
import os
import platform
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.1.9"
ASSETS = {
    ("linux", "x86_64"): ("linux-x86_64", ".so", "b959baa1d8dc88861b1edb337b8587178cdcb12d60b4998f9d10b6a82052d5d7"),
    ("linux", "aarch64"): ("linux-aarch64", ".so", "ea03d39541e478fab5974253c461e1cb5d77742f69e40cf96e3fad5bc309a37c"),
    ("darwin", "x86_64"): ("macos-x86_64", ".dylib", "53ad76e400786515e2edcaed2f01271dda846316390b761fadbd2dcf56aa4713"),
    ("darwin", "aarch64"): ("macos-aarch64", ".dylib", "8282126333399ddfe98bbbcc7a1936e7252625aac49df056a98be602e46bfd29"),
    ("windows", "x86_64"): ("windows-x86_64", ".dll", "51581189d52066b4dfc6631f6d7a3eab7dedc2260656ab09ca97ab3fb8165983"),
}


def validate(path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    connection.enable_load_extension(True)
    try:
        connection.load_extension(str(path))
    finally:
        connection.enable_load_extension(False)
    actual_version = connection.execute("SELECT vec_version()").fetchone()[0]
    connection.close()
    if actual_version != f"v{VERSION}":
        raise SystemExit(f"sqlite-vec version mismatch: expected v{VERSION}, got {actual_version}")


def main() -> None:
    if platform.system() == "Linux":
        libc = subprocess.run(["ldd", "--version"], capture_output=True, text=True, check=False)
        if "musl" in (libc.stdout + libc.stderr).lower():
            subprocess.run(
                [Path(__file__).with_name("build-sqlite-vec.sh")],
                env={**os.environ, "PYTHON": sys.executable},
                check=True,
            )
            return

    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    key = (platform.system().lower(), machine)
    try:
        platform_name, suffix, expected_hash = ASSETS[key]
    except KeyError as exc:
        raise SystemExit(f"sqlite-vec {VERSION} has no supported artifact for {key[0]}/{key[1]}") from exc

    asset = f"sqlite-vec-{VERSION}-loadable-{platform_name}.tar.gz"
    url = f"https://github.com/asg017/sqlite-vec/releases/download/v{VERSION}/{asset}"
    output = Path(__file__).resolve().parents[1] / "src" / "memu" / "database" / "sqlite" / f"vec0{suffix}"

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / asset
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise SystemExit(f"sqlite-vec checksum mismatch: expected {expected_hash}, got {actual_hash}")

        candidate = temporary_path / f"vec0{suffix}"
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember(candidate.name)
            source = bundle.extractfile(member)
            if source is None:
                raise SystemExit(f"sqlite-vec archive is missing {candidate.name}")
            candidate.write_bytes(source.read())
        candidate.chmod(0o755)

        subprocess.run([sys.executable, Path(__file__).resolve(), "--validate", candidate], check=True)
        os.replace(candidate, output)

    print(f"installed {output} (v{VERSION})")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--validate":
        validate(Path(sys.argv[2]))
    else:
        main()
