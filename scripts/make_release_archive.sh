#!/bin/sh
# Build one self-contained source archive (TreeMPU + the mpu_generators submodule, tracked files only).
# Usage: scripts/make_release_archive.sh [output.tar.gz]   (run from the repository root)
set -e
out="${1:-TreeMPU-$(python3 -c 'import re;print(re.search(r"version = \"(.*?)\"", open("pyproject.toml").read()).group(1))').tar.gz}"
tmp="$(mktemp -d)"
git archive --format=tar HEAD | tar -x -C "$tmp"
(cd mpu_generators && git archive --format=tar HEAD) | tar -x -C "$tmp/mpu_generators"
tar -czf "$out" -C "$tmp" .
rm -rf "$tmp"
echo "wrote $out"
