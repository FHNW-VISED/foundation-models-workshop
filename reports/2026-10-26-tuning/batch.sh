#!/bin/bash
cd "$(dirname "$0")"
for e in "dec_full 1500" "dec_scrib 1500" "dec_dino_only_full 1500" "rgb_only_full 1500" "unet_full 3000" "unet_scrib 3000" "dec_full20 3000" "unet_full20 6000"; do
  echo "== $e"
  uv run --project ~/Work/notebooks/foundation-models-workshop python run.py $e 2>&1 | grep -v Warning | tail -4
done
