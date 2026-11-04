#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
# Build the extension from THIS checkout; do not accidentally import another project.
(cd GroundingDINO && python -m pip install --no-build-isolation --no-deps -e .)
