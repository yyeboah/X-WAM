#!/usr/bin/env bash

set -euo pipefail

function setup_logging(){
    if [[ "${MASTER_RUNNER_LOGGING:-1}" == "0" ]]; then
        return 0
    fi

    local log_dir="${MASTER_RUNNER_LOG_DIR:-logs}"
    mkdir -p "${log_dir}"

    if [[ -z "${MASTER_RUNNER_LOG_FILE:-}" ]]; then
        MASTER_RUNNER_LOG_FILE="${log_dir}/master_runner_$(date +%Y%m%d_%H%M%S).log"
    fi

    export MASTER_RUNNER_LOG_FILE
    exec > >(tee -a "${MASTER_RUNNER_LOG_FILE}") 2>&1
    echo "Logging master_runner output to ${MASTER_RUNNER_LOG_FILE}"
}

function setup_wan22(){
    local dest="${WAN22_DIR:-checkpoints/wan22_5b}"
    local source_dir="${1:-${WAN22_SOURCE_DIR:-}}"
    local repo="${WAN22_REPO:-Wan-AI/Wan2.2-TI2V-5B}"

    mkdir -p "${dest}"

    if [[ -n "${source_dir}" ]]; then
        if [[ ! -d "${source_dir}" ]]; then
            echo "Wan2.2 source directory does not exist: ${source_dir}" >&2
            return 1
        fi

        cp -a "${source_dir}/." "${dest}/"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${repo}" --local-dir "${dest}"
    elif command -v hf >/dev/null 2>&1; then
        hf download "${repo}" --local-dir "${dest}"
    else
        echo "Install huggingface_hub or pass a local source directory:" >&2
        echo "  pip install -U huggingface_hub" >&2
        echo "  ./master_runner.sh setup_wan22 /path/to/Wan2.2-TI2V-5B" >&2
        return 1
    fi

    local missing=0
    for file in Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth; do
        if [[ ! -f "${dest}/${file}" ]]; then
            echo "Missing expected Wan2.2 file: ${dest}/${file}" >&2
            missing=1
        fi
    done

    if [[ "${missing}" -ne 0 ]]; then
        echo "Wan2.2 setup finished, but expected files are missing. Check ${dest}." >&2
        return 1
    fi

    echo "Wan2.2 base weights are ready at ${dest}"
}

function require_path(){
    local path="$1"
    local message="$2"

    if [[ ! -e "${path}" ]]; then
        echo "${message}" >&2
        return 1
    fi
}

function gpu_diagnostics(){
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
    echo
    echo "NVIDIA device nodes:"
    ls -la /dev/nvidia* 2>/dev/null || true
    echo
    echo "nvidia-smi:"
    nvidia-smi -L 2>&1 || true
    echo
    echo "PyTorch CUDA:"
    python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_build={torch.version.cuda}")
print(f"is_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
PY
}

function setup_robotwin_dataset_link(){
    local source_dir="${1:-datasets/RoboTwin}"
    local dest_dir="${2:-sft_datasets/RoboTwin}"

    if [[ ! -d "${source_dir}" ]]; then
        echo "RoboTwin dataset source does not exist: ${source_dir}" >&2
        return 1
    fi

    mkdir -p "$(dirname "${dest_dir}")"
    if [[ -e "${dest_dir}" && ! -L "${dest_dir}" ]]; then
        echo "Destination already exists and is not a symlink: ${dest_dir}" >&2
        return 1
    fi

    ln -sfn "../${source_dir}" "${dest_dir}"
    echo "RoboTwin dataset link is ready: ${dest_dir} -> ../${source_dir}"
}

function download_robotwin_missing_media(){
    local local_dir="${ROBOTWIN_DIR:-datasets/RoboTwin}"
    local workers="${ROBOTWIN_DOWNLOAD_WORKERS:-4}"

    python scripts/download_hf_stable.py sharinka0715/X-WAM-RoboTwin \
        --repo-type dataset \
        --local-dir "${local_dir}" \
        --include "video/head_camera/*" \
        --include "video/left_camera/*" \
        --include "video/right_camera/*" \
        --include "depth/right_camera/*" \
        --max-workers "${workers}"
}

function check_robotwin_media_layout(){
    local dataset_root="${1:-sft_datasets/RoboTwin}"
    local missing=0
    local expected=27500

    for path in \
        "${dataset_root}/video/head_camera" \
        "${dataset_root}/video/left_camera" \
        "${dataset_root}/video/right_camera" \
        "${dataset_root}/depth/head_camera" \
        "${dataset_root}/depth/left_camera" \
        "${dataset_root}/depth/right_camera"; do
        if [[ ! -d "${path}" ]]; then
            echo "Missing RoboTwin media directory: ${path}" >&2
            missing=1
            continue
        fi

        local count
        count="$(find "${path}" -type f -name 'episode_*.mp4' | wc -l)"
        if [[ "${count}" -lt "${expected}" ]]; then
            echo "Incomplete RoboTwin media directory: ${path} has ${count}/${expected} mp4 files" >&2
            missing=1
        fi
    done

    if [[ "${missing}" -ne 0 ]]; then
        echo "Resume missing RoboTwin media with: ./master_runner.sh download_robotwin_missing_media" >&2
        return 1
    fi
}

function preflight_training(){
    local missing=0

    require_path "checkpoints/wan22_5b/Wan2.2_VAE.pth" \
        "Missing Wan2.2 weights. Run: ./master_runner.sh setup_wan22" || missing=1
    require_path "checkpoints/wan22_5b/models_t5_umt5-xxl-enc-bf16.pth" \
        "Missing Wan2.2 T5 weights. Run: ./master_runner.sh setup_wan22" || missing=1
    require_path "checkpoints/pretrained/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt" \
        "Missing X-WAM pretrained checkpoint. Run: hf download sharinka0715/X-WAM-checkpoints --local-dir checkpoints" || missing=1
    require_path "sft_datasets/RoboTwin/data" \
        "Missing training dataset at sft_datasets/RoboTwin. Run: ./master_runner.sh setup_robotwin_dataset_link" || missing=1
    check_robotwin_media_layout "sft_datasets/RoboTwin" || missing=1

    local gpu_count
    gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
    if [[ "${gpu_count}" -lt 1 ]]; then
        echo "No CUDA GPUs are visible to PyTorch. Check NVIDIA driver, CUDA, or container GPU access." >&2
        missing=1
    fi

    python - <<'PY' || missing=1
import sys

try:
    import tensorboard  # noqa: F401
except ModuleNotFoundError:
    try:
        import tensorboardX  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "Missing TensorBoard logger dependency for "
            f"{sys.executable}. Run: python -m pip install tensorboard"
        )
PY

    if [[ "${missing}" -ne 0 ]]; then
        return 1
    fi
}

function run_training(){
    preflight_training
    local nproc_per_node="${NPROC_PER_NODE:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}"
    local master_port="${MASTER_PORT:-29500}"

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node="${nproc_per_node}" \
    --master_addr=localhost \
    --master_port="${master_port}" \
    scripts/train_sft.py \
    dataset=robotwin \
    exp_name=robotwin_sft \
    num_warmup_steps=400 \
    num_training_steps=40000
}

function usage(){
    cat <<'EOF'
Usage: ./master_runner.sh [function] [args...]

Functions:
  run_training              Start RoboTwin SFT training
  gpu_diagnostics           Log GPU/NVIDIA/PyTorch CUDA visibility
  setup_wan22 [source_dir]  Prepare checkpoints/wan22_5b
  download_robotwin_missing_media
                            Resume missing RoboTwin RGB/right-depth media
  setup_robotwin_dataset_link [source] [dest]
                            Link downloaded RoboTwin data into sft_datasets/

Wan2.2 setup options:
  WAN22_DIR=/path/to/dest
  WAN22_REPO=hf/repo-id
  WAN22_SOURCE_DIR=/path/to/downloaded/Wan2.2-TI2V-5B

Training options:
  NPROC_PER_NODE=8
  MASTER_PORT=29500

Download options:
  ROBOTWIN_DIR=datasets/RoboTwin
  ROBOTWIN_DOWNLOAD_WORKERS=4

Examples:
  ./master_runner.sh setup_wan22
  ./master_runner.sh setup_wan22 /path/to/Wan2.2-TI2V-5B
  ./master_runner.sh setup_robotwin_dataset_link
  ./master_runner.sh download_robotwin_missing_media
  NPROC_PER_NODE=4 ./master_runner.sh run_training
EOF
}

function main(){
    setup_logging

    local command="${1:-run_training}"
    if [[ "$#" -gt 0 ]]; then
        shift
    fi

    case "${command}" in
        run_training|setup_wan22|setup_robotwin_dataset_link|preflight_training|download_robotwin_missing_media)
            "${command}" "$@"
            ;;
        gpu_diagnostics)
            gpu_diagnostics
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "Unknown function: ${command}" >&2
            usage >&2
            return 1
            ;;
    esac
}

main "$@"
