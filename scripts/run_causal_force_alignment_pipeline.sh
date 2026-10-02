#!/usr/bin/env bash

# Collect deployable no-contact baselines, fit one causal force contract,
# replay all contact trajectories with that frozen contract, and rebuild the
# paired aligned datasets.  Each Isaac Sim trajectory runs in a fresh process.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

conda_env="${CONDA_ENV:-env_isaaclab}"
direction_test="${DIRECTION_TEST:-outputs/traj0_verified_visual_link7_force_smooth/traj_0/link7_frame_calibration/result.json}"
no_contact_root="${NO_CONTACT_ROOT:-outputs/causal_force_no_contact}"
contact_root="${CONTACT_ROOT:-outputs/causal_force_replay}"
alignment_root="${ALIGNMENT_ROOT:-outputs/force_alignment}"
alignment_config="$alignment_root/force_alignment_config.json"

if [[ ! -s "$direction_test" ]] || \
   ! jq -e '.summary.passed == true and .summary.sign_consistent == true and
             .incoming_wrench_read_body == "panda_link7" and
             ([.directions[].cross_axis_leakage_ratio] | max) <= 0.15' \
      "$direction_test" >/dev/null; then
    echo "Missing or invalid <=15% panda_link7 direction test: $direction_test" >&2
    exit 1
fi

mkdir -p -- "$no_contact_root" "$contact_root" "$alignment_root"

physics_args=(
    --task-kp-scale "${TASK_KP_SCALE:-1.0}"
    --task-kd-scale "${TASK_KD_SCALE:-1.0}"
)
append_optional_arg() {
    local flag="$1"
    local value="$2"
    if [[ -n "$value" ]]; then
        physics_args+=("$flag" "$value")
    fi
}
append_optional_arg --contact-offset-m "${CONTACT_OFFSET_M:-}"
append_optional_arg --held-static-friction "${HELD_STATIC_FRICTION:-}"
append_optional_arg --held-dynamic-friction "${HELD_DYNAMIC_FRICTION:-}"
append_optional_arg --fixed-static-friction "${FIXED_STATIC_FRICTION:-}"
append_optional_arg --fixed-dynamic-friction "${FIXED_DYNAMIC_FRICTION:-}"
append_optional_arg --solver-position-iterations "${SOLVER_POSITION_ITERATIONS:-}"
append_optional_arg --solver-velocity-iterations "${SOLVER_VELOCITY_ITERATIONS:-}"

current_stage=""
cleanup_stage() {
    if [[ "$current_stage" == "$no_contact_root"/.traj_*.stage.* ]] || \
       [[ "$current_stage" == "$contact_root"/.traj_*.stage.* ]]; then
        if [[ -d "$current_stage" ]]; then
            rm -rf -- "$current_stage"
        fi
    fi
}
trap cleanup_stage EXIT

for trajectory_id in $(seq 0 39); do
    final_dir="$no_contact_root/traj_$trajectory_id"
    if [[ -s "$final_dir/panda_link7_force_alignment_120hz.csv" ]] && \
       [[ -s "$final_dir/panda_link7_joint_state_120hz.csv" ]]; then
        echo "[NO-CONTACT $trajectory_id/39] already complete"
        continue
    fi
    if [[ -e "$final_dir" ]]; then
        echo "Refusing to overwrite incomplete no-contact output: $final_dir" >&2
        exit 1
    fi
    current_stage="$(mktemp -d "$no_contact_root/.traj_${trajectory_id}.stage.XXXXXX")"
    conda run --no-capture-output -n "$conda_env" \
        python -u scripts/replay_real_ee_cartesian_ppo.py \
        --headless --enable_cameras --device cuda:0 \
        --h5 "real_data/traj_$trajectory_id/data.h5" \
        --output-dir "$current_stage" \
        --link7-frame-calibration "$direction_test" \
        --link7-force-gain 1.0 \
        --link7-cutoff-hz 0.35 \
        --disable-hole-collision \
        "${physics_args[@]}"
    mv -- "$current_stage" "$final_dir"
    current_stage=""
done

echo "[FIT] q/qd baselines and post-filter real AR(1) residual"
conda run --no-capture-output -n "$conda_env" \
    python scripts/fit_causal_force_alignment.py \
    --real-data-dir real_data \
    --free-space-intervals real_data/free_space_intervals.json \
    --sim-no-contact-root "$no_contact_root" \
    --direction-test "$direction_test" \
    --output "$alignment_config" \
    --cutoff-hz 0.35 \
    --tare-seconds 3.0 \
    --output-hz 10.0

for trajectory_id in $(seq 0 39); do
    final_dir="$contact_root/traj_$trajectory_id"
    if [[ -s "$final_dir/data.h5" ]] && \
       [[ -s "$final_dir/panda_link7_force_model.csv" ]]; then
        echo "[CONTACT $trajectory_id/39] already complete"
        continue
    fi
    if [[ -e "$final_dir" ]]; then
        echo "Refusing to overwrite incomplete contact output: $final_dir" >&2
        exit 1
    fi
    current_stage="$(mktemp -d "$contact_root/.traj_${trajectory_id}.stage.XXXXXX")"
    real_h5="real_data/traj_$trajectory_id/data.h5"
    conda run --no-capture-output -n "$conda_env" \
        python -u scripts/replay_real_ee_cartesian_ppo.py \
        --headless --enable_cameras --device cuda:0 \
        --h5 "$real_h5" \
        --output-dir "$current_stage" \
        --link7-frame-calibration "$direction_test" \
        --force-alignment-config "$alignment_config" \
        --add-force-model-noise \
        --force-noise-seed "$trajectory_id" \
        --link7-force-gain 1.0 \
        --record-contact-pair \
        "${physics_args[@]}"
    conda run --no-capture-output -n "$conda_env" \
        python scripts/package_continuous_ppo_traj0.py \
        --real-h5 "$real_h5" \
        --sim-dir "$current_stage" \
        --force-dir "$current_stage"
    mv -- "$current_stage" "$final_dir"
    current_stage=""
done

echo "[BUILD] paired real/sim datasets"
conda run --no-capture-output -n "$conda_env" \
    python real-data-aligned/build_aligned_real.py \
    --force-alignment-config "$alignment_config"
conda run --no-capture-output -n "$conda_env" \
    python sim-data-aligned/build_aligned_sim.py \
    --source-root "$contact_root" \
    --visual-root sim-data \
    --output-root sim-data-aligned

echo "[EVAL] validation trajectories 32..39 and strict traj0 holdout"
conda run --no-capture-output -n "$conda_env" \
    python scripts/evaluate_causal_force_alignment.py \
    --real-root real-data-aligned \
    --sim-root sim-data-aligned \
    --geometry-root sim-data \
    --output "$alignment_root/evaluation.json"

echo "Done:"
echo "  config: $alignment_config"
echo "  evaluation: $alignment_root/evaluation.json"
echo "  contact source: $contact_root"
