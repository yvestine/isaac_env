#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
env_name="${CONDA_ENV_NAME:-env_isaaclab}"
isaaclab_tag="v3.0.0-beta2"
isaaclab_commit="28a37cecdd433c22d9eabd6a5954add9f13a8951"
isaaclab_dir="${ISAACLAB_DIR:-$repo_dir/.deps/IsaacLab}"

for command in conda git; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "[setup] missing command: $command" >&2
        exit 1
    fi
done
if ! git lfs version >/dev/null 2>&1; then
    echo "[setup] Git LFS is required. Install git-lfs and rerun this script." >&2
    exit 1
fi
if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "yes" ]]; then
    echo "[setup] NVIDIA Omniverse EULA acceptance is required." >&2
    echo "[setup] Rerun with: OMNI_KIT_ACCEPT_EULA=yes bash scripts/setup_environment.sh" >&2
    exit 1
fi

cd -- "$repo_dir"
git lfs install --local
git lfs pull

if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    conda env update --name "$env_name" --file environment.yml
else
    conda env create --name "$env_name" --file environment.yml
fi

conda run --no-capture-output -n "$env_name" \
    python -m pip install --upgrade "pip<27" "setuptools<82"
conda run --no-capture-output -n "$env_name" \
    python -m pip install --upgrade \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
conda run --no-capture-output -n "$env_name" \
    python -m pip install --upgrade \
    "isaacsim[all,extscache]==6.0.0.0" \
    --extra-index-url https://pypi.nvidia.com

if [[ ! -d "$isaaclab_dir/.git" ]]; then
    mkdir -p -- "$(dirname -- "$isaaclab_dir")"
    git clone --branch "$isaaclab_tag" --depth 1 \
        https://github.com/isaac-sim/IsaacLab.git "$isaaclab_dir"
fi
actual_commit="$(git -C "$isaaclab_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$isaaclab_commit" ]]; then
    git -C "$isaaclab_dir" fetch --depth 1 origin "$isaaclab_commit"
    git -C "$isaaclab_dir" checkout --detach "$isaaclab_commit"
fi

conda run --no-capture-output -n "$env_name" \
    bash "$isaaclab_dir/isaaclab.sh" -i "rl[rl-games],visualizer[kit]"
conda run --no-capture-output -n "$env_name" \
    python -m pip install --upgrade -r requirements-runtime.txt
conda run --no-capture-output -n "$env_name" \
    python -m pip install --editable source/tacex \
    --editable source/tacex_assets \
    --editable source/tacex_tasks

conda run --no-capture-output -n "$env_name" \
    python scripts/check_repository.py --check-environment

echo "[setup] ready: conda activate $env_name"
