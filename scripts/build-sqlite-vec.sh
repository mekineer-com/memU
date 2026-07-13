#!/bin/sh
set -eu

version=0.1.9
sha256=3acd67cb4aff080c7050926fd3cf8227905fe5b7ee3829d8ee5024ab1283cf61
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output="$root/src/memu/database/sqlite/vec0.so"
python=${PYTHON:-python3}
tmp=$(mktemp -d)
candidate_dir=
cleanup() {
    rm -rf "$tmp"
    [ -z "$candidate_dir" ] || rm -rf "$candidate_dir"
}
trap cleanup EXIT HUP INT TERM

archive="$tmp/sqlite-vec.tar.gz"
curl -fsSL "https://github.com/asg017/sqlite-vec/releases/download/v$version/sqlite-vec-$version-amalgamation.tar.gz" -o "$archive"
printf '%s  %s\n' "$sha256" "$archive" | sha256sum -c -
tar -xzf "$archive" -C "$tmp"

set -- -O3 -fPIC -shared
if ldd --version 2>&1 | grep -q musl; then
    set -- "$@" -Du_int8_t=uint8_t -Du_int16_t=uint16_t -Du_int64_t=uint64_t
fi

output_dir=$(dirname -- "$output")
mkdir -p "$output_dir"
candidate_dir=$(mktemp -d "$output_dir/.sqlite-vec.XXXXXX")
candidate="$candidate_dir/vec0.so"
cc "$@" "$tmp/sqlite-vec.c" -o "$candidate"

"$python" - "$candidate" "$version" <<'PY'
import sqlite3
import sys

path, expected = sys.argv[1:]
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
db.load_extension(path)
db.enable_load_extension(False)
actual = db.execute("SELECT vec_version()").fetchone()[0]
if actual != f"v{expected}":
    raise SystemExit(f"sqlite-vec version mismatch: expected v{expected}, got {actual}")
vector_type, length = db.execute(
    "SELECT vec_type(vec_f32('[1,2]')), vec_length(vec_f32('[1,2]'))"
).fetchone()
if (vector_type, length) != ("float32", 2):
    raise SystemExit(f"sqlite-vec vector check failed: {(vector_type, length)!r}")
nearest = db.execute(
    """
    WITH candidates(id, embedding) AS (
        VALUES ('near', vec_f32('[1,0]')), ('far', vec_f32('[0,1]'))
    )
    SELECT id
    FROM candidates
    ORDER BY vec_distance_cosine(embedding, vec_f32('[0.9,0.1]'))
    LIMIT 1
    """
).fetchone()[0]
if nearest != "near":
    raise SystemExit(f"sqlite-vec cosine check failed: nearest={nearest!r}")
PY

mv -f "$candidate" "$output"
rm -rf "$candidate_dir"
candidate_dir=
printf 'built %s (v%s)\n' "$output" "$version"
