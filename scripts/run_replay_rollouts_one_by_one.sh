#!/usr/bin/env bash

# Run each trajectory in a fresh Isaac Sim/conda process.
# Usage:
#   bash scripts/run_replay_rollouts_one_by_one.sh [start] [end] [output_root]
# Example (continue the interrupted batch):
#   bash scripts/run_replay_rollouts_one_by_one.sh 4 39

set -u

start_traj="${1:-0}"
end_traj="${2:-39}"
output_root="${3:-outputs/paired_rollouts_40_dr}"
conda_env="${CONDA_ENV:-env_isaaclab}"

if ! [[ "$start_traj" =~ ^[0-9]+$ && "$end_traj" =~ ^[0-9]+$ ]]; then
    echo "start/end must be non-negative integers" >&2
    exit 2
fi
if (( start_traj > end_traj )); then
    echo "start must be <= end" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

replay_pattern='[s]cripts/replay_real_joint_ppo.py'
active_group_pid=""

cleanup_active_process() {
    local group_pid="${active_group_pid:-}"
    if [[ -z "$group_pid" ]]; then
        return
    fi

    # The replay is launched in its own session/process group.  This cleanup
    # is scoped to that group, so unrelated Python training jobs are untouched.
    kill -TERM -- "-${group_pid}" 2>/dev/null || true
    for _ in {1..10}; do
        if ! kill -0 -- "-${group_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    kill -KILL -- "-${group_pid}" 2>/dev/null || true
    active_group_pid=""
}

handle_interrupt() {
    echo "[BatchReplay] interrupted; cleaning the active Isaac Sim process" >&2
    cleanup_active_process
    exit 130
}

trap cleanup_active_process EXIT
trap handle_interrupt INT TERM

for ((i = start_traj; i <= end_traj; i++)); do
    h5_path="real_data/traj_${i}/data.h5"
    output_dir="${output_root}/traj_${i}"

    if [[ ! -f "$h5_path" ]]; then
        echo "[BatchReplay] missing H5: $h5_path" >&2
        exit 1
    fi

    # A completed directory is skipped, while an empty/partial directory is
    # replayed from scratch by the standalone command below.
    if [[ -s "$output_dir/data.h5" \
        && -s "$output_dir/front_camera.mp4" \
        && -s "$output_dir/wrist_camera.mp4" \
        && -s "$output_dir/replay_metadata.json" ]]; then
        echo "[BatchReplay] skip completed traj_${i}"
        continue
    fi

    running="$(pgrep -af "$replay_pattern" || true)"
    if [[ -n "$running" ]]; then
        echo "[BatchReplay] another replay process is still running:" >&2
        echo "$running" >&2
        exit 2
    fi

    mkdir -p -- "$output_dir"
    echo "[BatchReplay] starting traj_${i} in a fresh Isaac Sim process"

    # Each trajectory gets a new session/process group.  Wait for it to finish
    # before explicitly cleaning that group and starting the next trajectory.
    setsid --wait conda run --no-capture-output -n "$conda_env" \
        python -u scripts/replay_real_joint_ppo.py \
        --enable_cameras \
        --domain-randomization \
        --dr-seed "$i" \
        --gripper-constant 0.0865 \
        --hole-reference gt-final-xy-fixed-z \
        --resolve-asset-collisions \
        --collision-substeps 4 \
        --h5 "$h5_path" \
        --output-dir "$output_dir" &
    active_group_pid=$!
    wait "$active_group_pid"
    status=$?
    cleanup_active_process
    if (( status != 0 )); then
        echo "[BatchReplay] traj_${i} failed with exit code ${status}; stopping" >&2
        exit "$status"
    fi

    running="$(pgrep -af "$replay_pattern" || true)"
    if [[ -n "$running" ]]; then
        echo "[BatchReplay] replay process remained after traj_${i}:" >&2
        echo "$running" >&2
        exit 2
    fi
    echo "[BatchReplay] finished traj_${i}"
done

echo "[BatchReplay] done: traj_${start_traj}..traj_${end_traj}"
