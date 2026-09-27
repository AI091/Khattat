#!/usr/bin/env bash
# Build diffvg into the project environment on CUDA 12/13, CMake 4 and recent GCC.
# Patches: newer pybind11, C++17, Python header path, and a GCC that nvcc accepts.
#
# Usage: scripts/build_diffvg.sh [--cpu]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFVG_DIR="$REPO_ROOT/third_party/diffvg"
PYBIND11_REF="v2.13.6"

DIFFVG_CUDA=1
[[ "${1:-}" == "--cpu" ]] && DIFFVG_CUDA=0

VENV="${VIRTUAL_ENV:-$REPO_ROOT/.venv}"
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: no environment at $VENV. Run 'uv sync --extra <cpu|cu128|cu130>' first." >&2
  exit 1
fi
echo "==> build tools"
uv pip install --quiet --python "$PY" cmake setuptools

if [[ ! -f "$DIFFVG_DIR/CMakeLists.txt" ]]; then
  echo "==> initialising diffvg submodule"
  git -C "$REPO_ROOT" submodule update --init --recursive third_party/diffvg
fi

echo "==> pybind11 $PYBIND11_REF"
git -C "$DIFFVG_DIR/pybind11" fetch --quiet --depth 1 origin "$PYBIND11_REF"
git -C "$DIFFVG_DIR/pybind11" checkout --quiet FETCH_HEAD

echo "==> C++17"
sed -i \
  -e 's/set(CMAKE_CUDA_STANDARD 11)/set(CMAKE_CUDA_STANDARD 17)/' \
  -e 's/-std=c++11/-std=c++17/g' \
  -e 's/PROPERTY CXX_STANDARD 11/PROPERTY CXX_STANDARD 17/' \
  "$DIFFVG_DIR/CMakeLists.txt"

echo "==> Python include path"
# FindPythonLibs reads PYTHON_INCLUDE_DIR, which setup.py does not pass.
grep -q "PYTHON_INCLUDE_DIR=" "$DIFFVG_DIR/setup.py" || sed -i \
  "s|'-DPYTHON_INCLUDE_PATH=' + include_path\]|'-DPYTHON_INCLUDE_PATH=' + include_path, '-DPYTHON_INCLUDE_DIR=' + include_path]|" \
  "$DIFFVG_DIR/setup.py"
grep -q "PYTHON_INCLUDE_DIR=" "$DIFFVG_DIR/setup.py" || { echo "setup.py patch failed" >&2; exit 1; }

pick_host_gcc() {
  local nvcc_bin max_gcc
  nvcc_bin="$(command -v nvcc || true)"
  [[ -z "$nvcc_bin" ]] && return 0
  local hdr="$(dirname "$nvcc_bin")/../targets/x86_64-linux/include/crt/host_config.h"
  [[ -f "$hdr" ]] || return 0
  max_gcc="$(grep -oP '__GNUC__ > \K[0-9]+' "$hdr" | head -1)"
  [[ -z "$max_gcc" ]] && return 0

  local current
  current="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
  if (( current <= max_gcc )); then
    return 0  # default gcc is fine
  fi
  for v in $(seq "$max_gcc" -1 9); do
    if command -v "gcc-$v" >/dev/null && command -v "g++-$v" >/dev/null; then
      echo "==> nvcc supports GCC <= $max_gcc, default is $current; using gcc-$v" >&2
      export CC="gcc-$v" CXX="g++-$v" CUDAHOSTCXX="g++-$v"
      return 0
    fi
  done
  echo "warning: nvcc needs GCC <= $max_gcc but only GCC $current is installed." >&2
  echo "         Install an older gcc (Arch: pacman -S gcc${max_gcc})." >&2
}
[[ "$DIFFVG_CUDA" == "1" ]] && pick_host_gcc

export CMAKE_POLICY_VERSION_MINIMUM=3.5
export DIFFVG_CUDA

echo "==> building diffvg (CUDA=$DIFFVG_CUDA)"
cd "$DIFFVG_DIR"
rm -rf build
"$PY" setup.py install

echo "==> verifying"
cd "$REPO_ROOT"
"$PY" - <<'EOF'
import torch, pydiffvg
pydiffvg.set_use_gpu(torch.cuda.is_available())
pts = torch.tensor([[50., 50.], [200., 30.], [250., 200.], [60., 220.]],
                   device=pydiffvg.get_device(), requires_grad=True)
path = pydiffvg.Path(num_control_points=torch.tensor([2]), points=pts, is_closed=True)
group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                            fill_color=torch.tensor([0., 0., 0., 1.]))
args = pydiffvg.RenderFunction.serialize_scene(256, 256, [path], [group])
img = pydiffvg.RenderFunction.apply(256, 256, 2, 2, 0, None, *args)
img[:, :, 3].sum().backward()
assert pts.grad is not None and pts.grad.norm() > 0, "no gradient reached control points"
print(f"diffvg OK  (gpu={pydiffvg.get_use_gpu()}, grad_norm={pts.grad.norm():.1f})")
EOF
