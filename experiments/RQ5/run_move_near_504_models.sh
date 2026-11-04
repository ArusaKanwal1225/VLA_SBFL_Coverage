#!/usr/bin/env bash
# Build, evaluate and remove each model environment sequentially.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=$(cd -- "${DEEPHYPERION_ROOT:-/home/ubuntu/Deephyperion_VLA}" && pwd -P)
export DEEPHYPERION_ROOT="$PROJECT_DIR"
SETUP_DIR="$SCRIPT_DIR/environment_mount"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

usage() {
  cat <<'HELP'
Usage: bash run_move_near_504_models.sh [options] [model ...]
Models: pi0 gr00t eo1 spatialvla-4b (default: all four)
Options:
  --resume          Resume existing episode results with the same configuration.
  --video           Save videos (must match the option used for an existing run).
  --seed NUMBER     Simulator seed (default: 2024).
  --output PATH     Results root (default: ../results/move_near_504).
  -h, --help        Show this help.

Existing environments and checkpoints are reused. Resources are removed only
after all 504 tasks for that model complete successfully; failures preserve them.
HELP
}

models=()
runner_args=()
while (($#)); do
  case "$1" in
    pi0|gr00t|eo1|spatialvla-4b) models+=("$1"); shift ;;
    --resume|--video) runner_args+=("$1"); shift ;;
    --seed|--output)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      if [[ "$1" == --output ]]; then
        # Resolve relative to the caller, before changing directory.
        output_path=$(realpath -m -- "$2")
        runner_args+=("$1" "$output_path")
      else
        runner_args+=("$1" "$2")
      fi
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option or model: $1" >&2; usage >&2; exit 2 ;;
  esac
done
((${#models[@]})) || models=(pi0 gr00t eo1 spatialvla-4b)
declare -A seen=()
for model in "${models[@]}"; do
  [[ ! ${seen[$model]+yes} ]] || { echo "Duplicate model: $model" >&2; exit 2; }
  seen[$model]=1
done

if ! command -v conda >/dev/null 2>&1; then
  [[ -x /opt/conda/bin/conda ]] || { echo 'Conda not found in PATH.' >&2; exit 1; }
  export PATH="/opt/conda/bin:$PATH"
fi
CONDA_BASE=$(conda info --base)
[[ -x "$CONDA_BASE/bin/python" ]] || { echo 'Conda base Python not found.' >&2; exit 1; }
command -v xvfb-run >/dev/null || { echo 'xvfb-run is required.' >&2; exit 1; }
[[ -f "$SCRIPT_DIR/run_move_near_504.py" ]] || { echo 'Runner not found.' >&2; exit 1; }

owned_env=''
owned_checkpoints=()
cleanup() {
  local failed=0 item
  if [[ -n "$owned_env" ]]; then
    echo "Removing Conda environment: $owned_env"
    conda env remove --name "$owned_env" --yes || failed=1
  fi
  for item in "${owned_checkpoints[@]}"; do
    # Names are a fixed allowlist and must remain direct children of checkpoints.
    [[ "$item" == "$CHECKPOINT_DIR/"* && "${item#"$CHECKPOINT_DIR/"}" != */* ]] || { echo "Unsafe checkpoint path: $item" >&2; failed=1; continue; }
    if [[ -e "$item" || -L "$item" ]]; then
      echo "Removing downloaded checkpoint: $item"
      rm -rf -- "$item" || failed=1
    fi
  done
  owned_env=''
  owned_checkpoints=()
  return "$failed"
}
on_exit() {
  local status=$?
  trap - EXIT
  if ((status != 0)) && [[ -n "$owned_env" ]]; then
    echo "Run failed or was interrupted; preserving environment $owned_env and its checkpoints." >&2
    echo "Retry with the same options and --resume." >&2
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for model in "${models[@]}"; do
  case "$model" in
    pi0) env_name=PI0; setup=PI0.sh; checkpoint_names=(lerobot-pi0-bridge lerobot-pi0-fractal) ;;
    gr00t) env_name=GR00T; setup=GR00T.sh; checkpoint_names=(GR00T-N1.6-bridge GR00T-N1.6-fractal) ;;
    eo1) env_name=EO1; setup=EO1.sh; checkpoint_names=(eo1-qwen25_vl-fractal eo1-qwen25_vl-bridge) ;;
    spatialvla-4b) env_name=SPATIALVLA; setup=SPATIALVLA.sh; checkpoint_names=(spatialvla-4b-mix-224-pt) ;;
  esac
  [[ -f "$SETUP_DIR/$setup" ]] || { echo "Missing setup script: $setup" >&2; exit 1; }
  env_json=$(conda env list --json)
  env_exists=0
  if "$CONDA_BASE/bin/python" -c 'import json,os,sys; sys.exit(0 if any(os.path.basename(p)==sys.argv[1] for p in json.load(sys.stdin)["envs"]) else 1)' "$env_name" <<< "$env_json"; then
    env_exists=1
  fi
  owned_env="$env_name"
  owned_checkpoints=()
  for name in "${checkpoint_names[@]}"; do owned_checkpoints+=("$CHECKPOINT_DIR/$name"); done
  if ((env_exists)); then
    echo "=== Reusing existing environment $env_name ==="
    for checkpoint in "${owned_checkpoints[@]}"; do
      if [[ ! -f "$checkpoint/config.json" ]]; then
        if [[ "$model" == spatialvla-4b ]]; then
          echo "=== Continuing the partial SpatialVLA installation ==="
          (cd -- "$SETUP_DIR" && SPATIALVLA_RESUME=1 bash "$setup")
          break
        fi
        echo "Checkpoint missing or incomplete: $checkpoint. The existing environment and files are preserved." >&2
        exit 1
      fi
    done
  else
    for checkpoint in "${owned_checkpoints[@]}"; do
      if [[ -e "$checkpoint" || -L "$checkpoint" ]]; then
        echo "Checkpoint exists but environment $env_name is missing: $checkpoint. Preserving it." >&2
        exit 1
      fi
    done
    echo "=== Building $model ($env_name) ==="
    (cd -- "$SETUP_DIR" && bash "$setup")
  fi
  # Later dependency installations can replace setuptools. Restore the version
  # used by the project's model installers, then verify the required module.
  echo "=== Ensuring setuptools 80.10.2 and pkg_resources in $env_name ==="
  conda run --no-capture-output -n "$env_name" python3.10 -m pip install 'setuptools==80.10.2'
  conda run --no-capture-output -n "$env_name" python3.10 -c 'import importlib.metadata as m; import pkg_resources; assert m.version("setuptools") == "80.10.2"; print("setuptools", m.version("setuptools"), "and pkg_resources available")'
  if [[ "$model" == pi0 ]]; then
    echo "=== Ensuring compatible PI0 dataset dependencies ==="
    conda run --no-capture-output -n "$env_name" python3.10 -m pip install 'datasets==3.6.0' 'huggingface-hub==0.36.2' 'numpy==1.24.4'
    conda run --no-capture-output -n "$env_name" python3.10 -c 'import datasets; from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy; print("PI0 imports OK")'
  fi
  echo "=== Running the 504 tasks for $model ==="
  (cd -- "$PROJECT_DIR/experiments" && MUJOCO_GL=glx xvfb-run -s '-screen 0 1280x720x24' -a \
    conda run --no-capture-output -n "$env_name" python3.10 "$SCRIPT_DIR/run_move_near_504.py" --model "$model" "${runner_args[@]}")
  cleanup || { echo "Cleanup failed for $model." >&2; exit 1; }
  echo "=== Finished $model ==="
done
