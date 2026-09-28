#!/usr/bin/env bash
set -euo pipefail

python examples/demo_audioldm2.py \
  --prompt "rain falling, followed by a crash of thunder" \
  --seed "[1688]" \
  --guidance_scale 3.5 \
  --audio_length_in_s 10.24 \
  --n_timesteps 50 \
  --algo_version energy \
  --correction_mode correct-tweedie \
  --num_ts_to_correct "[5]" \
  --num_latent_corrector_steps 1 \
  --init_latent_corrector_steps 10 \
  --output_dir outputs/audioldm2
