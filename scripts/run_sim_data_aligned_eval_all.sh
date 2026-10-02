#!/usr/bin/env bash
set -euo pipefail

# Run one TAVLA configuration over a contiguous set of the 40 real-data profiles.
# Usage: bash scripts/run_sim_data_aligned_eval_all.sh [start] [end] [output_root]

start_profile="${1:-0}"
end_profile="${2:-39}"
output_root="${3:-outputs/sim_data_force_trend_eval}"
conda_env="${CONDA_ENV:-env_isaaclab}"
server_host="${POLICY_HOST:-114.214.164.36}"
server_port="${POLICY_PORT:-8000}"
action_start_index="${TAVLA_ACTION_START_INDEX:-1}"
replan_actions="${REPLAN_ACTIONS:-5}"
teacher_hold_steps="${TEACHER_HOLD_STEPS:-1}"
episode_length_s="${EPISODE_LENGTH_S:-60}"
eval_mode="${EVAL_MODE:-overwrite}"
bundle_root="${OPENPI_ROOT:-sim_side_test_bundle}"
wrench_adapter="${WRENCH_ADAPTER_PATH:-$bundle_root/assets/wrench_adapters/sim_aligned_to_real_affine.pt}"
force_config="${FORCE_TREND_CONFIG_PATH:-$bundle_root/configs/tavla_sim_force_trend_affine.json}"

if ! [[ "$start_profile" =~ ^[0-9]+$ && "$end_profile" =~ ^[0-9]+$ ]]; then
    echo "start/end must be non-negative integers" >&2
    exit 2
fi
if (( start_profile > end_profile || end_profile > 39 )); then
    echo "profile range must satisfy 0 <= start <= end <= 39" >&2
    exit 2
fi
if [[ "$server_port" != "8000" && "$server_port" != "8001" ]]; then
    echo "POLICY_PORT must be 8000 or 8001" >&2
    exit 2
fi
if [[ "$action_start_index" != "1" && "$action_start_index" != "5" ]]; then
    echo "TAVLA_ACTION_START_INDEX must be 1 or 5" >&2
    exit 2
fi
if [[ "$replan_actions" != "5" && "$replan_actions" != "10" ]]; then
    echo "REPLAN_ACTIONS must be 5 or 10" >&2
    exit 2
fi
if [[ "$eval_mode" != "overwrite" && "$eval_mode" != "resume" ]]; then
    echo "EVAL_MODE must be overwrite or resume" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

if [[ ! -s "$wrench_adapter" || ! -s "$force_config" ]]; then
    echo "Missing force-trend bundle under $bundle_root" >&2
    exit 1
fi
conda run -n "$conda_env" python scripts/check_repository.py

active_group_pid=""
cleanup_active_process() {
    local group_pid="${active_group_pid:-}"
    if [[ -z "$group_pid" ]]; then
        return
    fi
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
trap cleanup_active_process EXIT INT TERM

for ((profile_id = start_profile; profile_id <= end_profile; profile_id++)); do
    profile_name="$(printf 'profile_%02d' "$profile_id")"
    output_dir="${output_root}/${profile_name}"
    if [[ "$eval_mode" == "resume" && -s "$output_dir/summary.json" && -s "$output_dir/episodes.csv" ]]; then
        echo "[SimDataAlignedTAVLA] skip completed ${profile_name}"
        continue
    fi

    overwrite_args=()
    if [[ "$eval_mode" == "overwrite" ]]; then
        overwrite_args+=(--overwrite-output)
    fi
    echo "[SimDataAlignedTAVLA] ${profile_name} port=${server_port} start=${action_start_index} steps=${replan_actions}"

    setsid --wait conda run --no-capture-output -n "$conda_env" \
        env TERM=xterm-256color \
        ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/pi0_randomized_eval.py \
        --policy tavla \
        --sim-data-profile-id "$profile_id" \
        --tavla-host "$server_host" \
        --tavla-port "$server_port" \
        --tavla-action-start-index "$action_start_index" \
        --replan-actions "$replan_actions" \
        --teacher-hold-steps "$teacher_hold_steps" \
        --episodes 1 \
        --episode-length-s "$episode_length_s" \
        --sim-data-dir sim-data \
        --sim-data-aligned-dir sim-data-aligned \
        --openpi-root "$bundle_root" \
        --wrench-adapter "$wrench_adapter" \
        --force-trend-config "$force_config" \
        --output-dir "$output_dir" \
        --flat-output \
        "${overwrite_args[@]}" \
        --save-policy-input-video \
        --device cuda \
        --headless &
    active_group_pid=$!
    set +e
    wait "$active_group_pid"
    status=$?
    set -e
    active_group_pid=""
    if (( status != 0 )); then
        echo "[SimDataAlignedTAVLA] ${profile_name} failed with exit code ${status}" >&2
        exit "$status"
    fi
    if [[ ! -s "$output_dir/summary.json" || ! -s "$output_dir/episodes.csv" ]]; then
        echo "[SimDataAlignedTAVLA] ${profile_name} completed without summary files" >&2
        exit 1
    fi
done

echo "[SimDataAlignedTAVLA] done: profile_${start_profile}..profile_${end_profile}"
