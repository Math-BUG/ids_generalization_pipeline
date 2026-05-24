#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv not found in $PROJECT_DIR" >&2
  exit 1
fi

python - <<'PY'
import sys

try:
    import cupy
    import cuml
except Exception as exc:
    print("RAPIDS GPU backend is NOT available.")
    print(f"Import error: {exc}")
    sys.exit(1)

print("RAPIDS GPU backend is available.")
print("CuPy:", cupy.__version__)
print("cuML:", cuml.__version__)

try:
    import cudf

    print("cuDF:", cudf.__version__)
except Exception:
    print("cuDF: not installed (not required by the current pipeline)")

try:
    n_devices = cupy.cuda.runtime.getDeviceCount()
    print("CUDA devices:", n_devices)
    for device_id in range(n_devices):
        props = cupy.cuda.runtime.getDeviceProperties(device_id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"GPU {device_id}: {name}")
except Exception as exc:
    print(f"Could not query CUDA devices: {exc}")
    sys.exit(1)
PY
