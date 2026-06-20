#!/usr/bin/env bash

set -euo pipefail

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

    if [[ "${missing}" -ne 0 ]]; then
        return 1
    fi
}

function run_training(){
    preflight_training

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
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
  setup_wan22 [source_dir]  Prepare checkpoints/wan22_5b
  setup_robotwin_dataset_link [source] [dest]
                            Link downloaded RoboTwin data into sft_datasets/

Wan2.2 setup options:
  WAN22_DIR=/path/to/dest
  WAN22_REPO=hf/repo-id
  WAN22_SOURCE_DIR=/path/to/downloaded/Wan2.2-TI2V-5B

Examples:
  ./master_runner.sh setup_wan22
  ./master_runner.sh setup_wan22 /path/to/Wan2.2-TI2V-5B
  ./master_runner.sh setup_robotwin_dataset_link
EOF
}

function main(){
    local command="${1:-run_training}"
    if [[ "$#" -gt 0 ]]; then
        shift
    fi

    case "${command}" in
        run_training|setup_wan22|setup_robotwin_dataset_link|preflight_training)
            "${command}" "$@"
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
