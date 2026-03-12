#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${ROOT_DIR}/.conda/watersic"
SPEC="${ROOT_DIR}/configs/env/watersic_conda.yml"

export HF_HOME="${ROOT_DIR}/outputs/hf_cache"
export HUGGINGFACE_HUB_CACHE="${ROOT_DIR}/outputs/hf_cache/hub"
export TRANSFORMERS_CACHE="${ROOT_DIR}/outputs/hf_cache/transformers"
export PIP_CACHE_DIR="${ROOT_DIR}/.cache/pip"

mkdir -p "${ROOT_DIR}/.conda" "${ROOT_DIR}/.cache/pip" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ ! -d "${PREFIX}" ]]; then
  conda env create --prefix "${PREFIX}" --file "${SPEC}"
else
  conda env update --prefix "${PREFIX}" --file "${SPEC}" --prune
fi

cat <<EOF
Environment ready at:
  ${PREFIX}

Activate with:
  conda activate ${PREFIX}

Cache directories:
  HF_HOME=${HF_HOME}
  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}
  TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
EOF
