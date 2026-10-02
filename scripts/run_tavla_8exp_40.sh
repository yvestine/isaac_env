#!/usr/bin/env bash
set -euo pipefail

# 2 models x 2 action starts x 2 chunk lengths x 40 in-distribution profiles.
# Usage: bash scripts/run_tavla_8exp_40.sh [output_root] [overwrite|resume]

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
output_root="${1:-outputs/tavla_8exp}"
mode="${2:-overwrite}"

if [[ "$mode" != "overwrite" && "$mode" != "resume" ]]; then
    echo "mode must be overwrite or resume" >&2
    exit 2
fi

cd -- "$repo_dir"
for port in 8000 8001; do
    for action_start in 1 5; do
        for replan_actions in 5 10; do
            experiment="port_${port}_start_${action_start}_steps_${replan_actions}"
            echo "[TAVLA-8EXP] ${experiment}"
            POLICY_PORT="$port" \
            TAVLA_ACTION_START_INDEX="$action_start" \
            REPLAN_ACTIONS="$replan_actions" \
            EVAL_MODE="$mode" \
                bash scripts/run_sim_data_aligned_eval_all.sh \
                0 39 "$output_root/$experiment"
        done
    done
done

python scripts/summarize_tavla_8exp.py "$output_root" --strict
