#!/usr/bin/env bash
set -euo pipefail

case_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$case_dir/../.." && pwd)
setup_src=${SETUP_FILE:-input/setup_2t.txt}
bin=${COREFL_2T_BIN:-"$repo_dir/corefl-2t"}

cd "$case_dir"
python3 generate_mesh.py
cp "$setup_src" input/setup.txt
rm -rf output history.dat "Man, we are Finished.txt"

if [[ ! -x "$bin" ]]; then
  echo "missing executable: $bin" >&2
  echo "build the 2T binary first or set COREFL_2T_BIN" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec "$bin"
