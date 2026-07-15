#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-python}"
SCENE_PATH="${ROOT_DIR}/scenes/humanoid_loading_factory.usd"

cd "${ROOT_DIR}"

if [[ ! -f "${SCENE_PATH}" ]]; then
    "${ISAAC_PYTHON}" scripts/create_factory_scene.py --headless
fi

exec "${ISAAC_PYTHON}" scripts/preview_agv_pipeline_demo.py "$@"

