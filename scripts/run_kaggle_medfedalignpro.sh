#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/FedALign-MedFedAlignPro}"
DOMAINS="${DOMAINS:-nih,guangzhou}"
ROUND="${ROUND:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HELDOUT_DOMAIN="${HELDOUT_DOMAIN:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${REPO_DIR}"

python -m pip install -q -r requirements.txt

CMD=(
  python main.py MedFedAlignPro
  -d medical_cxr
  --domains "${DOMAINS}"
  --round "${ROUND}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${HELDOUT_DOMAIN}" ]]; then
  CMD+=(--heldout_domain "${HELDOUT_DOMAIN}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARRAY[@]}")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
