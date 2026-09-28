#!/usr/bin/env bash
set -euo pipefail

python examples/demo_sdxl.py \
  --prompt "a hairy shark and two spotted clams" \
  --seed "[1688]" \
  --guidance_scale 5.0 \
  --n_timesteps 50 \
  --algo_version energy \
  --correction_mode correct-tweedie \
  --num_ts_to_correct "[5]" \
  --num_latent_corrector_steps 1 \
  --init_latent_corrector_steps 10 \
  --output_dir outputs/sdxl
