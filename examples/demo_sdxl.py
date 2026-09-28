import argparse
import ast
import gc
import hashlib
import math
import os
import random
import sys
import torch.utils.checkpoint as torch_checkpoint
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import en_core_web_trf
import matplotlib.pyplot as plt
import torch
import yaml
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from tqdm import tqdm
from transformers import logging as hf_logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tilt.text_utils import (
    encode_prompt,
    extract_base_nouns,
    get_concept_indices,
    remove_adjectives,
    remove_articles_from_beginning,
    remove_conjunctions,
    remove_conjunctions_from_beginning,
    remove_wh_words,
)


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch for repeatable demo runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def force_nonreentrant_checkpoint():
    """Force torch.utils.checkpoint to use ``use_reentrant=False`` inside the block.

    diffusers 0.29.2 enables reentrant gradient checkpointing, which does not
    support double-backward (needed by the curl diagnostic) and is less
    memory-efficient. Non-reentrant checkpointing supports both first- and
    second-order backprop through the chained UNet passes of the energy corrector.
    """
    orig = torch_checkpoint.checkpoint

    def patched(*args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return orig(*args, **kwargs)

    torch_checkpoint.checkpoint = patched
    try:
        yield
    finally:
        torch_checkpoint.checkpoint = orig


@dataclass
class DemoConfig:
    prompt: str = "a hairy shark and two spotted clams"
    prompt_orig: str = "a hairy shark and two spotted clams"
    concept: str = ""
    negative_prompt: str = ""
    seed: int = 0
    device: str = "cuda:0"
    output_path_all: str = "./outputs/sdxl/"
    guidance_scale: float = 5.0
    n_timesteps: int = 50
    sd_version: str = "xl"
    perform_latent_correction: bool = True
    num_ts_to_correct: int | list[int] = 10
    num_latent_corrector_steps: int = 1
    init_latent_corrector_steps: int = 10
    nts_boosted_grad: int = 2
    nts_to_init_correct: int = 2
    x0_hat_score_source: str = "multi-cfg"
    correction_mode: str = "reverse-sde"
    # Per-phase correction modes for the hybrid algos (hybrid-1-2 / energy-hybrid-1-2).
    # None => fall back to correction_mode.
    algo1_correction_mode: str | None = None
    algo2_correction_mode: str | None = None
    eta: float = 0.3
    eta_schedule_factor: float = 2.0
    beta: float = 1.0
    kappa: float = 0.1
    dps_t_eps: int = 300
    resolution_h: int = 1024
    resolution_w: int = 1024
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0
    latent_channels: int = 4
    use_attribute_guidance: bool = False
    use_cfgpp: bool = False
    gamma: float = 0.0
    algo_version: str = "option1"
    num_steps_first_algo: int = 2
    eta_opt1: float | None = None
    beta_opt1: float | None = None
    eta_opt2: float | None = None
    beta_opt2: float | None = None
    psi: float = 0.3
    save_intermediates: bool = False
    run_mode: str = "debug"
    run_id: int = 0
    ddim_forward_type: str = "multi"
    x0_interm_noise_type: str = "multi-cfg"
    x0_final_interm_noise_type: str = "multi-cfg"
    update_step_type: str = "split_step"
    # --- scalar-energy reward guidance (algo_version="energy") ---
    energy_num_samples: int = 2
    energy_tau_min: int = 200
    energy_tau_max: int = 500
    energy_weight_type: str = "uniform"
    energy_use_cfg: bool = False
    run_curl_diagnostic: bool = False
    curl_num_dirs: int = 4
    # --- adaptive per-concept weights for the energy reward (scratch4.tex §1.3) ---
    energy_adaptive_weights: bool = False
    energy_adaptive_zeta: float = 1.0
    energy_adaptive_norm: str = "max"
    # --- Euclidean net-gradient corrector (algo_version="energy-netgrad") ---
    netgrad_gamma: float = 1.0
    netgrad_crn: bool = False


def compute_adaptive_concept_weights(d, zeta, norm_type="max", eps=1e-8):
    """Softmin simplex weights over per-concept score-difference norms (scratch4.tex Eq. 8).
    

        w_i = exp(-ζ · s_i) / Σ_j exp(-ζ · s_j),    s = normalized d

    `d` is the [K] vector of score-difference norms
        d_i(x0) = E_{τ,ε}[ ω(τ) ‖ε_θ(x_τ,τ,C) - ε_θ(x_τ,τ,c_i)‖² ]     (Eq. 7)
    """
    d = d.detach().flatten().to(dtype=torch.float32)
    K = d.numel()
    if K < 2 or float(zeta) == 0.0:
        return torch.full_like(d, 1.0 / max(K, 1))

    if norm_type == "max":
        s = d / (d.abs().max() + eps)
    elif norm_type == "mean":
        s = d / (d.mean().abs() + eps)
    elif norm_type == "none":
        s = d
    else:
        raise ValueError(f"Unknown adaptive weight norm_type: {norm_type}")

    return torch.softmax(-float(zeta) * s, dim=0)


def get_model_card(sd_version: str) -> str:
    if sd_version != "xl":
        raise ValueError(f"Unsupported SDXL debug version: {sd_version}")
    return "stabilityai/stable-diffusion-xl-base-1.0"


@torch.no_grad()
def get_text_embeds(
    prompts,
    neg_prompts,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
):
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders=[text_encoder, text_encoder_2],
        tokenizers=[tokenizer, tokenizer_2],
        prompt=prompts,
        text_input_ids_list=None,
    )
    uncond_embeds, pooled_uncond_embeds = encode_prompt(
        text_encoders=[text_encoder, text_encoder_2],
        tokenizers=[tokenizer, tokenizer_2],
        prompt=neg_prompts,
        text_input_ids_list=None,
    )
    return (
        torch.cat([uncond_embeds, prompt_embeds]),
        torch.cat([pooled_uncond_embeds, pooled_prompt_embeds]),
    )


def compute_time_ids(config: DemoConfig) -> torch.Tensor:
    original_size = (config.resolution_h, config.resolution_w)
    target_size = (config.resolution_h, config.resolution_w)
    crops_coords_top_left = (config.crops_coords_top_left_h, config.crops_coords_top_left_w)
    return torch.tensor([list(original_size + crops_coords_top_left + target_size)])


def process_concepts(
    concepts,
    nlp,
    remove_wh=True,
    remove_conj_beg=True,
    remove_conj=False,
    remove_art_beg=False,
    remove_adj=False,
):
    if remove_wh:
        concepts = remove_wh_words(concepts, nlp)
    if remove_conj_beg:
        concepts = remove_conjunctions_from_beginning(concepts, nlp)
    if remove_conj:
        concepts = remove_conjunctions(concepts, nlp)
    if remove_art_beg:
        concepts = remove_articles_from_beginning(concepts, nlp)
    if remove_adj:
        concepts = remove_adjectives(concepts, nlp)
    return concepts


def build_prompt_layout(config: DemoConfig, tokenizer, nlp):
    prompt_orig = config.prompt_orig.lower()
    prompt_items = [c.strip() for c in extract_base_nouns(prompt_orig, nlp) if c.strip()]
    if config.concept.strip():
        concept_items = [c.strip() for c in config.concept.split("+") if c.strip()]
    else:
        doc = nlp(prompt_orig)
        prompt_anchor = [
            chunk.text
            for chunk in doc.noun_chunks
            if chunk.text not in ["top", "the side", "the left", "the right"]
        ]
        concept_items = [
            c.strip()
            for c in process_concepts(
                prompt_anchor,
                nlp,
                remove_wh=True,
                remove_conj_beg=True,
                remove_conj=False,
                remove_art_beg=False,
                remove_adj=False,
            )
            if c.strip()
        ]

    prompt_sep_items = prompt_items + [prompt_orig]
    concept_prompt_items = concept_items + [prompt_orig]
    prompt_config = SimpleNamespace(
        prompt_orig=prompt_orig,
        prompt="+".join(prompt_sep_items),
        concept="+".join(concept_prompt_items),
        negative_prompt=config.negative_prompt,
        use_attribute_guidance=False,
    )
    prompt_helper = SimpleNamespace(tokenizer=tokenizer)
    prompt_helper.prompt_orig = prompt_config.prompt_orig.split("+")[0]
    prompt_helper.prompt_sep = prompt_config.prompt.split("+")
    prompt_helper.concepts = prompt_config.concept.split("+")
    prompt_helper.prompts = [prompt_helper.prompt_orig] + prompt_helper.prompt_sep
    prompt_helper.concept_indices_list = get_concept_indices(
        tokenizer, prompt_helper.prompt_orig, prompt_helper.concepts
    )

    all_prompts = list(prompt_helper.prompts)
    if config.use_attribute_guidance and concept_items:
        all_prompts.extend(concept_items)

    base_concept_start = 2
    base_concept_end = base_concept_start + len(prompt_items)
    attribute_start = base_concept_end + 1

    return {
        "prompt_orig": prompt_orig,
        "prompt_items": prompt_items,
        "concept_items": concept_items,
        "prompt_sep_items": prompt_sep_items,
        "concept_prompt_items": concept_prompt_items,
        "all_prompts": all_prompts,
        "base_concept_slice": slice(base_concept_start, base_concept_end),
        "attribute_slice": slice(attribute_start, attribute_start + len(concept_items)),
    }
def stratified_tau(tau_lo, tau_hi, n_samples, k):
    """One jittered-grid draw of tau: bin k of n_samples, spanning [tau_lo, tau_hi].

    The reward is an expectation over tau estimated from energy_num_samples draws, and
    iid draws clump by chance. That matters here because the energy varies enormously
    across the window -- E_C measures ~2.6x larger at tau=433 than at tau=578 -- so a
    large share of the estimator's variance is simply which taus came up. Splitting the
    window into n equal bins and drawing one tau uniformly inside each keeps the
    estimator unbiased while removing the clumping component.

    Degenerates to a plain uniform draw at n_samples == 1, so single-particle runs are
    unaffected. Consumes one value from the global torch RNG per call, exactly as the
    torch.randint it replaces did.
    """
    span = (tau_hi + 1 - tau_lo) / n_samples
    lo = tau_lo + k * span
    return min(int(lo + torch.rand(1).item() * span), tau_hi)

def main():
    parser = argparse.ArgumentParser(description="TILT demo for SDXL.")
    parser.add_argument("--prompt", type=str, default="a hairy shark and two spotted clams")
    parser.add_argument("--concept", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=str, default="[5994, 1688]")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./outputs/sdxl/")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--n_timesteps", type=int, default=50)
    parser.add_argument("--correction_mode", choices=["dps-style", "reverse-sde", "correct-tweedie"], default="correct-tweedie", help="Correction mode: dps-style | reverse-sde | correct-tweedie")
    parser.add_argument("--algo1_correction_mode", choices=["dps-style", "reverse-sde", "correct-tweedie"], default=None, help="Correction mode for the FIRST phase of hybrid-1-2 / energy-hybrid-1-2 (steps < --num_steps_first_algo). Defaults to --correction_mode.")
    parser.add_argument("--algo2_correction_mode", choices=["dps-style", "reverse-sde", "correct-tweedie"], default=None, help="Correction mode for the SECOND phase of hybrid-1-2 / energy-hybrid-1-2 (steps >= --num_steps_first_algo). Defaults to --correction_mode.")
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--eta_schedule_factor", type=float, default=2.0, help="Slope f of the linear per-step decay applied to eta during correction: eta_t = eta*(1 - f*step/n_timesteps). At the default f=2.0 the schedule reaches 0 at step n_timesteps/2 (step 25 of 50) and goes NEGATIVE past it, which pushes against the reward -- so it also caps the useful --num_ts_to_correct at n_timesteps/f. Lower it to correct further into the trajectory (f=1.0 -> zero at the last step); f=0.0 disables the decay and applies a constant eta.")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.3, help="interpolation coeff within reverse-sde correction, between noise_pred and grad_noise")
    parser.add_argument("--num_ts_to_correct", type=str, default="[5]")
    parser.add_argument("--num_latent_corrector_steps", type=int, default=1)
    parser.add_argument("--init_latent_corrector_steps", type=int, default=10)
    parser.add_argument("--nts_boosted_grad", type=int, default=2, help="Number of initial corrected timesteps whose correct-tweedie update replaces the Tweedie mean with a mostly-gradient mix instead of the additive eta-scaled nudge (x0/grad factors: energy 0.2/0.8 then 0.3/0.7; energy-opt2 0.3/0.7 then 0.4/0.6). correct-tweedie only.")
    parser.add_argument("--nts_to_init_correct", type=int, default=2, help="Number of initial timesteps that use init_latent_corrector_steps instead of num_latent_corrector_steps (typically 0, 1 or 2; 0 disables the init corrector steps).")
    parser.add_argument("--x0_hat_score_source", choices=["multi", "multi-cfg", "uncond"], default="multi")
    parser.add_argument("--x0_interm_noise_type", choices=["multi", "multi-cfg", "uncond"], default="multi-cfg")
    parser.add_argument("--x0_final_interm_noise_type", choices=["multi", "multi-cfg", "uncond"], default="multi-cfg", help="Noise type defining the Tweedie mean on the FINAL latent-corrector step (the outer update); --x0_interm_noise_type is used for the preceding corrector steps.")
    parser.add_argument("--update_step_type", choices=["split_step", "joint_step"], default="split_step", help="How the final outer update in reverse-sde correction is applied: split_step | joint_step")
    parser.add_argument("--dps_t_eps", type=int, default=300, help="Timestep at which x0 score is evaluated for DPS correction.")
    parser.add_argument("--ddim_forward_type", choices=["cfg", "uncond", "multi"], default="cfg", help="DDIM forward type: cfg | uncond | multi")
    parser.add_argument("--use_attribute_guidance", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.1, help="strength of attribute guidance")
    parser.add_argument("--disable_correction", action="store_true")
    parser.add_argument("--use_cfgpp", action="store_true")
    parser.add_argument("--psi", type=float, default=0.3, help="noise mixing coefficient for intermediate corrector steps in correct-tweedie mode")
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--algo_version", type=str, default="option1", help="algorithm version: option1 | option2 | mpgd | hybrid-1-2 | energy | energy-opt2 | energy-netgrad | energy-hybrid-1-2")
    parser.add_argument("--netgrad_gamma", type=float, default=1.0, help="Inner ascent step size gamma for algo_version=energy-netgrad (eta scales the accumulated displacement).")
    parser.add_argument("--netgrad_crn", action="store_true", help="energy-netgrad: share the reward's (tau, eps) draws across all inner ascent steps of a timestep, so the whole inner loop ascends ONE fixed landscape (makes the reward comparable/monotone across inner steps).")
    parser.add_argument("--energy_num_samples", type=int, default=2, help="Number of (tau, eps) draws for the scalar-energy reward (CRN averaged).")
    parser.add_argument("--energy_tau_min", type=int, default=200, help="Lower bound of the internal classifier noise band tau for the energy reward.")
    parser.add_argument("--energy_tau_max", type=int, default=500, help="Upper bound of the internal classifier noise band tau for the energy reward.")
    parser.add_argument("--energy_weight_type", choices=["uniform", "min-snr"], default="uniform", help="Per-tau weighting w(tau) for the energy reward.")
    parser.add_argument("--energy_use_cfg", action="store_true", help="Use CFG-combined noise in the energy reward (default: raw conditional).")
    parser.add_argument("--energy_adaptive_weights", action="store_true", help="Adaptive per-concept weights for the energy reward (scratch4.tex Eq. 9): replace the uniform 1/K by a softmin over the score-difference norms d_i, so the DOMINANT concept gets the largest weight. Applies to algo_version=energy only.")
    parser.add_argument("--energy_adaptive_zeta", type=float, default=1.0, help="Softmin temperature ζ for --energy_adaptive_weights (0 → uniform 1/K, i.e. the mean-reduced reward; larger → concentrates on argmin_i d_i).")
    parser.add_argument("--energy_adaptive_norm", choices=["max", "mean", "none"], default="max", help="How d_i is normalized before the softmin: max = d/max(d) (ratios in (0,1], near-ties stay near-uniform, ζ ~ 5-10); mean = d/mean(d); none = raw d (literal Eq. 8, needs ζ ~ 1e-4). A z-score variant was removed: at K=2 it depends only on the ORDER of d, not the gap.")
    parser.add_argument("--run_curl_diagnostic", action="store_true", help="Run the curl/antisymmetry diagnostic on the guidance field (debug + save_intermediates only).")
    parser.add_argument("--curl_num_dirs", type=int, default=4, help="Number of random direction pairs for the curl diagnostic.")
    parser.add_argument("--num_steps_first_algo", type=int, default=2, help="For hybrid-1-2 / energy-hybrid-1-2: number of correction steps (0-indexed) using option1 before switching to option2")
    parser.add_argument("--eta_opt1", type=float, default=None, help="For hybrid-1-2 / energy-hybrid-1-2: eta for the option1 phase (required for those algos)")
    parser.add_argument("--beta_opt1", type=float, default=None, help="For hybrid-1-2 / energy-hybrid-1-2: beta for the option1 phase (required for those algos)")
    parser.add_argument("--eta_opt2", type=float, default=None, help="For hybrid-1-2 / energy-hybrid-1-2: eta for the option2 phase (required for those algos)")
    parser.add_argument("--beta_opt2", type=float, default=None, help="For hybrid-1-2 / energy-hybrid-1-2: beta for the option2 phase (required for those algos)")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID for tracking experiments.")
    args = parser.parse_args()

    config = DemoConfig(
        prompt=args.prompt,
        prompt_orig=args.prompt,
        concept=args.concept,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        device=args.device,
        output_path_all=args.output_dir,
        use_cfgpp=args.use_cfgpp,
        guidance_scale=args.guidance_scale,
        n_timesteps=args.n_timesteps,
        correction_mode=args.correction_mode,
        algo1_correction_mode=args.algo1_correction_mode,
        algo2_correction_mode=args.algo2_correction_mode,
        eta=args.eta,
        eta_schedule_factor=args.eta_schedule_factor,
        beta=args.beta,
        kappa=args.kappa,
        num_ts_to_correct=args.num_ts_to_correct,
        num_latent_corrector_steps=args.num_latent_corrector_steps,
        init_latent_corrector_steps=args.init_latent_corrector_steps,
        nts_boosted_grad=args.nts_boosted_grad,
        nts_to_init_correct=args.nts_to_init_correct,
        x0_hat_score_source=args.x0_hat_score_source,
        x0_interm_noise_type=args.x0_interm_noise_type,
        x0_final_interm_noise_type=args.x0_final_interm_noise_type,
        update_step_type=args.update_step_type,
        dps_t_eps=args.dps_t_eps,
        use_attribute_guidance=args.use_attribute_guidance,
        perform_latent_correction=not args.disable_correction,
        gamma=args.gamma,
        algo_version=args.algo_version,
        num_steps_first_algo=args.num_steps_first_algo,
        eta_opt1=args.eta_opt1,
        beta_opt1=args.beta_opt1,
        eta_opt2=args.eta_opt2,
        beta_opt2=args.beta_opt2,
        psi=args.psi,
        save_intermediates=args.save_intermediates,
        ddim_forward_type=args.ddim_forward_type,
        run_id=args.run_id,
        energy_num_samples=args.energy_num_samples,
        energy_tau_min=args.energy_tau_min,
        energy_tau_max=args.energy_tau_max,
        energy_weight_type=args.energy_weight_type,
        energy_use_cfg=args.energy_use_cfg,
        run_curl_diagnostic=args.run_curl_diagnostic,
        curl_num_dirs=args.curl_num_dirs,
        energy_adaptive_weights=args.energy_adaptive_weights,
        energy_adaptive_zeta=args.energy_adaptive_zeta,
        energy_adaptive_norm=args.energy_adaptive_norm,
        netgrad_gamma=args.netgrad_gamma,
        netgrad_crn=args.netgrad_crn,
    )

    # ---------------------------- Checks ------------------------------------------------
    # Algos driven by the per-phase opt1/opt2 knobs instead of the global --beta/--eta.
    HYBRID_ALGOS = ("hybrid-1-2", "energy-hybrid-1-2")
    # Algos that evaluate the scalar-energy reward (so their run dir must carry energy_str()).
    ENERGY_ALGOS = ("energy", "energy-opt2", "energy-netgrad", "energy-hybrid-1-2")

    assert config.nts_to_init_correct >= 0, f"--nts_to_init_correct must be >= 0, got {config.nts_to_init_correct}"

    if config.algo_version in HYBRID_ALGOS:
        av = config.algo_version
        assert config.eta_opt1 is not None, f"--eta_opt1 must be set when using algo_version={av}"
        assert config.beta_opt1 is not None, f"--beta_opt1 must be set when using algo_version={av}"
        assert config.eta_opt2 is not None, f"--eta_opt2 must be set when using algo_version={av}"
        assert config.beta_opt2 is not None, f"--beta_opt2 must be set when using algo_version={av}"
        assert config.algo1_correction_mode is not None, f"--algo1_correction_mode must be set when using algo_version={av}"
        assert config.algo2_correction_mode is not None, f"--algo2_correction_mode must be set when using algo_version={av}"

    # Per-phase correction modes. The hybrid algos run two different correctors back to
    # back, so each phase gets its own mode; every other algo falls back to the single
    # --correction_mode. These are the values threaded into the update_x_with_* functions.
    algo1_correction_mode = config.algo1_correction_mode 
    algo2_correction_mode = config.algo2_correction_mode 

    if config.algo_version not in HYBRID_ALGOS and (
        config.algo1_correction_mode is not None or config.algo2_correction_mode is not None
    ):
        print(
            f"[WARN] --algo1_correction_mode/--algo2_correction_mode are only used by "
            f"{HYBRID_ALGOS}; ignoring them for algo_version={config.algo_version}."
        )

    # update_x_with_energy_grad_opt2 is only implemented/validated for reverse-sde, so every
    # path that routes into it must use that mode: --algo_version energy-opt2 (via the single
    # --correction_mode) and the 2nd phase of energy-hybrid-1-2 (via --algo2_correction_mode).

    # NOTE: commenting out the below constraint because correct-tweedie in energy-opt2 always use grad with jacobian
    # so the correct-tweedie mode is also valid. 

    # if config.algo_version == "energy-opt2":
    #     assert config.correction_mode == "reverse-sde", (
    #         f"algo_version=energy-opt2 requires --correction_mode reverse-sde, "
    #         f"got {config.correction_mode!r}"
    #     )
    # if config.algo_version == "energy-hybrid-1-2":
    #     assert algo2_correction_mode == "reverse-sde", (
    #         f"algo_version=energy-hybrid-1-2 runs energy-opt2 in its 2nd phase, which requires "
    #         f"reverse-sde; got algo2_correction_mode={algo2_correction_mode!r} "
    #         f"(from --algo2_correction_mode={config.algo2_correction_mode!r}, "
    #         f"--correction_mode={config.correction_mode!r})"
    #     )

    def corr_mode_str():
        """Run-dir tag for the correction mode(s); hybrids carry both phases."""
        if config.algo_version in HYBRID_ALGOS:
            return f"corrMode-{algo1_correction_mode}-{algo2_correction_mode}_"
        return f"corrMode-{config.correction_mode}_"

    def eta_beta_str():
        if config.algo_version in HYBRID_ALGOS:
            # num_steps_first_algo is in here too: it is the knob unique to the hybrids, so a
            # switch-point sweep would otherwise write every variant to the same path.
            return (
                f"beta_opt1{config.beta_opt1}_eta_opt1{config.eta_opt1}"
                f"_beta_opt2{config.beta_opt2}_eta_opt2{config.eta_opt2}"
                f"_nfirst{config.num_steps_first_algo}"
            )
        return f"beta{config.beta}_eta{config.eta}"

    def adaptive_w_str():
        """Tag identifying the adaptive-weight setting; "" when it is off.

        Anything named per-run must carry this, otherwise a --energy_adaptive_zeta sweep
        writes every variant to the same path. Empty when adaptive weights are off, so
        uniform-reward runs keep their existing names.
        """
        if config.algo_version in ENERGY_ALGOS and config.energy_adaptive_weights:
            return f"_advw{config.energy_adaptive_zeta}{config.energy_adaptive_norm}"
        return ""

    def energy_str():
        if config.algo_version not in ENERGY_ALGOS:
            return ""
        s = (
            f"_nEsamp{config.energy_num_samples}"
            f"_tau{config.energy_tau_min}-{config.energy_tau_max}"
            f"_w{config.energy_weight_type}"
            f"_Ecfg{config.energy_use_cfg}"
            f"_crn{config.netgrad_crn}"
        )
        if config.algo_version == "energy-netgrad":
            s += f"_gam{config.netgrad_gamma}"
        s += adaptive_w_str()
        return s

    def nts_init_str():
        """Tag for --nts_to_init_correct; "" at its default of 2, which reproduces the old
        hardcoded behaviour, so existing run dirs keep their names."""
        return f"_ntsInit{config.nts_to_init_correct}" if config.nts_to_init_correct != 2 else ""

    def extra_knobs_str(include_psi=True):
        """Knobs that change the samples but were missing from the run-dir names.

        Two jobs differing only in --x0_final_interm_noise_type wrote to the same dir, and
        the second silently overwrote the first's samples and scores. Each knob is appended
        only when it differs from its argparse default, so default-valued runs keep their
        existing names (same convention as adaptive_w_str). Debug mode passes
        include_psi=False because its run-dir name always carries psi already.
        """
        s = nts_init_str()
        if config.x0_final_interm_noise_type != "multi-cfg":
            s += f"_x0fin{config.x0_final_interm_noise_type}"
        if include_psi and config.psi != 0.3:
            s += f"_psi{config.psi}"
        if config.update_step_type != "split_step":
            s += f"_upd{config.update_step_type}"
        if config.eta_schedule_factor != 2.0:
            s += f"_etaSched{config.eta_schedule_factor}"
        if config.use_attribute_guidance:
            s += f"_attrG{config.gamma}"
        if not config.perform_latent_correction:
            s += "_noCorr"
        if config.negative_prompt:
            s += "_neg" + hashlib.sha1(config.negative_prompt.encode()).hexdigest()[:6]
        return s

    def bounded_name(name, limit=240):
        """Keep a single path component under the filesystem's 255-byte limit.

        Hybrid run dirs are already ~223 bytes and the metrics log name adds ~25 more.
        Past 255 the shell `>` redirect fails and all four evaluators silently never run.
        An over-long name keeps a readable prefix plus a hash of the full name, so it
        stays unique.
        """
        if len(name) <= limit:
            return name
        return f"{name[:limit - 12]}_h{hashlib.sha1(name.encode()).hexdigest()[:10]}"

    def _parse_int_list(spec):
        if isinstance(spec, int):
            return [spec]
        if isinstance(spec, list):
            return [int(v) for v in spec]
        value = str(spec).strip()
        if not value:
            return []
        if value[0] in "[(":
            parsed = ast.literal_eval(value)
            if isinstance(parsed, int):
                return [parsed]
            return [int(v) for v in parsed]
        try:
            return [int(value)]
        except ValueError:
            return [int(v.strip()) for v in value.split(",") if v.strip()]
            
    def _fmt_stats(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        elif isinstance(v, tuple):
            return "(" + ", ".join(f"{e:.4f}" if isinstance(e, float) else str(e) for e in v) + ")"
        return str(v)
    

    viz_state = {"seed_dir": None, "seed": None}
    stats_history = {}

    hf_logging.set_verbosity_error()
    prompt_dir_name = config.prompt.replace(os.sep, "_").replace(" ", "_")

    model_card = get_model_card(config.sd_version)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    print(f"Loading SDXL model: {model_card}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_card,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_vae_slicing()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    ).to(device)
    unet = pipe.unet
    scheduler = DDIMScheduler.from_pretrained(model_card, subfolder="scheduler")
    scheduler.set_timesteps(config.n_timesteps, device=device)
    image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))

    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    add_time_ids = compute_time_ids(config).to(device)

    nlp = en_core_web_trf.load()

    if config.run_mode == "debug":
        prompt_layout = build_prompt_layout(config, tokenizer, nlp)
        all_prompts = prompt_layout["all_prompts"]
        text_embeds, text_embeds_pool = get_text_embeds(
            all_prompts,
            [config.negative_prompt],
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
        )
        text_embeds = text_embeds.to(device)
        text_embeds_pool = text_embeds_pool.to(device)
    else:
        # Reassigned per-prompt in the eval loop; update_x_* closures capture by reference.
        prompt_layout = None
        all_prompts = None
        text_embeds = None
        text_embeds_pool = None



    correction_counts = _parse_int_list(config.num_ts_to_correct)
    seed_values = _parse_int_list(config.seed)
    total_steps = len(scheduler.timesteps)
    for count in correction_counts:
        if count < 0 or count > total_steps:
            raise ValueError(f"Correction count {count} is out of range for n_timesteps={total_steps}.")

    def _predict_noise(x, t, text_embed, pool_embed):
        udtype = next(unet.parameters()).dtype
        x_input = x.to(dtype=udtype)
        text_embed_input = text_embed.to(device=x.device, dtype=udtype)
        pool_embed_input = pool_embed.to(device=x.device, dtype=udtype)
        added_conditions = {"time_ids": add_time_ids.repeat(text_embed_input.shape[0], 1)}
        added_conditions.update({"text_embeds": pool_embed_input})
        noise_pred = unet(
            x_input,
            t,
            encoder_hidden_states=text_embed_input,
            added_cond_kwargs=added_conditions,
        )["sample"]
        return noise_pred.to(dtype=torch.float32)

    def _noise2score(noise_pred, alpha_t):
        return -noise_pred / (1 - alpha_t).sqrt()
    def _score2noise(score, alpha_t):
        return -score * (1 - alpha_t).sqrt()
    def _x0_to_noise(x_t, x0_hat, alpha_t):
        return (x_t - alpha_t.sqrt() * x0_hat) / (1 - alpha_t).sqrt()

    def _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg):
        x0_noise_type = config.x0_final_interm_noise_type if cstep == num_latent_corrector_steps - 1 else config.x0_interm_noise_type
        if x0_noise_type == "multi-cfg":
            return noise_multi_cfg.detach()
        elif x0_noise_type == "multi":
            return noise_multi.detach()
        else:
            return noise_uncond.detach()

    def _compute_x0_from_xt(x, at, noise):
        return (x - (1 - at).sqrt() * noise) / at.sqrt()

    def _update_x_cur_from_x_tilde(
        x_cur, cstep, num_latent_corrector_steps, use_cfgpp, psi,
        at, at_prev, x_tilde_0, noise_uncond, noise_multi, forward_noise
    ):
        if cstep == num_latent_corrector_steps - 1:
            if use_cfgpp:
                return at_prev.sqrt() * x_tilde_0 + (1 - at_prev).sqrt() * noise_uncond.detach()
            else:
                return at_prev.sqrt() * x_tilde_0 + (1 - at_prev).sqrt() * forward_noise.detach()
        else:
            rand_noise = torch.randn_like(x_cur).detach()
            # rand_noise = rand_noise / (rand_noise.norm() + 1e-4) * noise_multi.norm()
            rand_noise = rand_noise / (rand_noise.abs().max() + 1e-3) * forward_noise.abs().max()
            psi_t = torch.ones_like(at) * psi
            #! Changing the intermediate noise to match with the denoised noise. 
            fwd_noise = psi_t * rand_noise.detach() + (1 - psi_t) * forward_noise.detach() #! some bug, why?  #TODO: simple convex? 
            fwd_noise = fwd_noise / (fwd_noise.abs().max() + 1e-3) * forward_noise.abs().max() # Needs length normalization. Sensitive to legth. 
            return at.sqrt() * x_tilde_0 + (1 - at).sqrt() * fwd_noise

    def _get_alphas(t):
        at = scheduler.alphas_cumprod[t.cpu()].to(dtype=torch.float32)
        t_scalar = int(t.item())
        ts_list = scheduler.timesteps.tolist()
        cur_idx = next((i for i, v in enumerate(ts_list) if int(v) == t_scalar), -1)
        if cur_idx == -1 or cur_idx == len(ts_list) - 1:
            at_prev = scheduler.final_alpha_cumprod.to(dtype=torch.float32)
        else:
            at_prev = scheduler.alphas_cumprod[int(ts_list[cur_idx + 1])].to(dtype=torch.float32)
        return at, at_prev

    def _get_x_t_prev(
        x_t, noise_pred_base, net_grad, t, at, at_prev, eta,
        guidance_scale, update_step_type, x0_interm_noise_type, use_cfgpp,
    ):
        noise_uncond = noise_pred_base[0:1]
        noise_multi = noise_pred_base[1:2]
        noise_cfg = noise_uncond + guidance_scale * (noise_multi - noise_uncond)

        # The noise that defines the Tweedie mean, per --x0_interm_noise_type.
        if x0_interm_noise_type == "multi-cfg":
            interm_noise = noise_cfg.detach()
        elif x0_interm_noise_type == "multi":
            interm_noise = noise_multi.detach()
        else:
            interm_noise = noise_uncond.detach()

        if update_step_type == "split_step":
            # net_grad applied in x_t-space, AFTER the DDIM step.
            x0_interm = (x_t.detach() - (1 - at).sqrt() * interm_noise) / at.sqrt()
            out_noise = noise_uncond.detach() if use_cfgpp else interm_noise
            x_interm = at_prev.sqrt() * x0_interm + (1 - at_prev).sqrt() * out_noise
            # return x_interm + float(eta) * net_grad
            netgrad_factor = at_prev.sqrt() / at.sqrt()  # This is the scaling factor to match the length of net_grad with x_interm
            x_prev = x_interm + netgrad_factor * net_grad #! net_grad's length should be  some fraction of (at_prev.sqrt() * x0_interm)
            return x_prev / (x_prev.abs().max() + 1e-3) * x_interm.abs().max()  #! length normalization to match the scale of x_interm

        elif update_step_type == "joint_step":
            # net_grad folded into the noise prediction BEFORE the step.
            updated_noise = interm_noise - float(eta) * net_grad  # -ve sign to match the noise_pred update direction
            updated_noise = updated_noise / (updated_noise.abs().max() + 1e-3) * interm_noise.abs().max()  #! length normalization to match the scale of interm_noise
            updated_tweedie = (x_t - (1 - at).sqrt() * updated_noise) / at.sqrt()
            out_noise = noise_uncond.detach() if use_cfgpp else updated_noise
            return at_prev.sqrt() * updated_tweedie + (1 - at_prev).sqrt() * out_noise

        else:
            raise ValueError(f"Unknown update_step_type: {update_step_type}")

    def update_x_with_dps(x, t, num_latent_corrector_steps, eta, beta, correction_mode=None):
        """`correction_mode` overrides config.correction_mode (per-phase hybrid modes)."""
        correction_mode = correction_mode or config.correction_mode
        text_embed_uncond = text_embeds[0:1]
        text_embed_multi = text_embeds[1:2]
        text_embed_uncond_pool = text_embeds_pool[0:1]
        text_embed_multi_pool = text_embeds_pool[1:2]

        base_slice = prompt_layout["base_concept_slice"]
        attr_slice = prompt_layout["attribute_slice"]
        concept_chunks = [text_embeds[base_slice]]
        concept_pool_chunks = [text_embeds_pool[base_slice]]
        if config.use_attribute_guidance and prompt_layout["concept_items"]:
            concept_chunks.append(text_embeds[attr_slice])
            concept_pool_chunks.append(text_embeds_pool[attr_slice])
        text_embed_concepts = torch.cat([chunk for chunk in concept_chunks if chunk.shape[0] > 0], dim=0)
        text_embed_concepts_pool = torch.cat(
            [chunk for chunk in concept_pool_chunks if chunk.shape[0] > 0],
            dim=0,
        )

        if text_embed_concepts.shape[0] == 0:
            text_embed_concepts = text_embed_multi
            text_embed_concepts_pool = text_embed_multi_pool

        at, at_prev = _get_alphas(t)
        dps_t_eps = t.new_tensor(config.dps_t_eps)
        at_eps = scheduler.alphas_cumprod[dps_t_eps.cpu()]

        x_cur = x.detach().to(dtype=torch.float32)
        stats = {}
        unet.requires_grad_(False)

        with torch.autocast(device_type="cuda", enabled=False), torch.enable_grad():
            for cstep in range(num_latent_corrector_steps):
                x_cur = x_cur.detach().requires_grad_(True)

                noise_pred_base = _predict_noise(
                    torch.cat([x_cur, x_cur]),
                    t,
                    torch.cat([text_embed_uncond, text_embed_multi], dim=0),
                    torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0),
                )
                if cstep == 0:
                    x_orig = x_cur.detach().clone()
                    noise_pred_base_orig = noise_pred_base.detach().clone()

                noise_uncond = noise_pred_base[0:1]
                noise_multi = noise_pred_base[1:2]
                noise_multi_cfg = noise_uncond + config.guidance_scale * (noise_multi - noise_uncond)

                if config.x0_hat_score_source == "multi-cfg":
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_multi_cfg) / at.sqrt()
                elif config.x0_hat_score_source == "multi":
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_multi) / at.sqrt()
                else:
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_uncond) / at.sqrt()

                z_noise = torch.randn_like(x0_hat).detach()
                x_eps_latent = at_eps.sqrt() * x0_hat + (1 - at_eps).sqrt() * z_noise
                with torch.no_grad():
                    noise_multi_x0 = _predict_noise(x_eps_latent.detach(), dps_t_eps,
                        text_embed_multi,
                        text_embed_multi_pool,
                    )
                    score_multi_x0 = _noise2score(noise_multi_x0, at_eps)
                    
                    K = text_embed_concepts.shape[0]
                    # print(K)
                    if not config.use_attribute_guidance:
                        score_concepts_sum = torch.zeros_like(score_multi_x0)
                        for cc in range(text_embed_concepts.shape[0]):
                            noise_concept_x0 = _predict_noise(x_eps_latent.detach(), dps_t_eps,
                                text_embed_concepts[cc : cc + 1],
                                text_embed_concepts_pool[cc : cc + 1],
                            )
                            score_concept_x0 = _noise2score(noise_concept_x0, at_eps)
                            score_concepts_sum += score_concept_x0
                        #! This is coming out to be very small in magnitude
                        composed_score_x0 = score_multi_x0 - (1 / K) * score_concepts_sum
                        # composed_score_x0 = config.beta * score_multi - config.beta * score_concepts_sum
                    else:
                        score_nouns_sum = torch.zeros_like(score_multi_x0)
                        score_concepts_sum = torch.zeros_like(score_multi_x0)
                        for cc in range(text_embed_concepts.shape[0]//2):
                            noise_noun = _predict_noise(
                                    x_eps_latent.detach(), dps_t_eps,
                                    text_embed_concepts[cc : cc + 1],
                                    text_embed_concepts_pool[cc : cc + 1],
                                )
                            score_nouns_sum += _noise2score(noise_noun, at_eps)
                        for cc in range(text_embed_concepts.shape[0]//2, text_embed_concepts.shape[0]):
                            noise_concept_x0 = _predict_noise(
                                    x_eps_latent.detach(), dps_t_eps,
                                    text_embed_concepts[cc : cc + 1],
                                    text_embed_concepts_pool[cc : cc + 1],
                                )
                            score_concepts_sum += _noise2score(noise_concept_x0, at_eps)
                            
                        composed_score_x0 = beta * score_multi_x0 \
                                        - (2+config.gamma) * (beta / text_embed_concepts.shape[0]) * score_nouns_sum \
                                        + config.gamma * (beta / text_embed_concepts.shape[0]) * score_concepts_sum

                # This grad is in score form
                grad =  torch.autograd.grad(
                    outputs=x0_hat,
                    inputs=x_cur,
                    grad_outputs=composed_score_x0,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0] * beta

                
                if correction_mode == "reverse-sde":
                    cgmin, cgmax = grad.min().item(), grad.max().item()

                    x_interm = x_cur.detach()  # stay at the same timestep

                    # correction_tensor = None #(1 - config.kappa) * noise_multi_cfg.detach() + config.kappa * _score2noise(grad, at).detach()
                    # correction_tensor = correction_tensor / (correction_tensor.norm() + 1e-3) * noise_multi.detach().norm()
                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    x_tilde_0 = (1 - eta / num_latent_corrector_steps) * x0_hat_interm + (eta / num_latent_corrector_steps) * grad.detach()

                    grad = grad.detach() / (grad.norm() + 1e-4) * x0_hat_interm.norm()  #! force normalize to avoid magnitude issues
                    x_cur = (x_interm + (float(eta) / num_latent_corrector_steps) * grad).detach()

                    # final outer update at the end of the loop
                    if cstep == num_latent_corrector_steps - 1:
                        net_grad = (x_cur - x_orig).detach()
                        x_cur = _get_x_t_prev(
                            x_t=x_orig,
                            noise_pred_base=noise_pred_base_orig,
                            net_grad=net_grad,
                            t=t,
                            at=at,
                            at_prev=at_prev,
                            eta=eta,
                            guidance_scale=config.guidance_scale,
                            update_step_type=config.update_step_type,
                            x0_interm_noise_type=config.x0_interm_noise_type,
                            use_cfgpp=config.use_cfgpp,
                        )

                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:
                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")

                        for _name, _tensor in [
                            ("x0_hat", x0_hat),
                            ("composed_score_x0", composed_score_x0),
                            ("grad", grad),
                            ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_cur", x_cur),
                            ("x_tilde_0", x_tilde_0),
                        ]:
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_raw.png")
                            viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val} k={cstep}")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_decoded.png")
                            decode_tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(decode_tensor, out_path)
                elif correction_mode == "correct-tweedie":
                    cgmin, cgmax = grad.min().item(), grad.max().item()
                    grad = grad.clamp(-5, 5) #! check approproiate value
                    
                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    snr = (1 - at).sqrt() / at.sqrt()  # can be used to schedule eta
                    x_tilde_0 = (1- eta) * x0_hat_interm + eta * grad.detach() #! make it convex comb. 

                    # Force-normalize to avoid magnitude drift
                    x_tilde_0 = x_tilde_0 / (x_tilde_0.norm() + 1e-4) * x0_hat_interm.norm()
                    correction_tensor = _x0_to_noise(x_cur.detach(), x_tilde_0, at)

                    # debug
                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:
                        
                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")
                        
                        for _name, _tensor in [
                            ("x0_hat", x0_hat.detach()),
                            ("score_concept_x0", score_concept_x0.detach()),
                            ("composed_score_x0", composed_score_x0),
                            ("grad", grad),
                            ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_tilde_0", x_tilde_0), 
                        ]:
                            if torch.isnan(_tensor).any() or torch.isinf(_tensor).any():
                                print(f"[WARN] {_name} has NaN/Inf")
                            _tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(_tensor, os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}.png"))
                    x_cur = _update_x_cur_from_x_tilde(
                        x_cur, cstep, num_latent_corrector_steps, config.use_cfgpp, config.psi,
                        at, at_prev, x_tilde_0, noise_uncond, noise_multi,
                        forward_noise=noise_multi_cfg, #! #noise_multi
                    )
                else:
                    x_intermediate = scheduler.step(noise_multi_cfg, t, x_cur.detach(), return_dict=False)[0]
                    correction_tensor = grad
                    x_cur = (x_intermediate + float(eta) * grad).detach()

                stats = {
                    "mode": correction_mode,
                    "correction_tensor": (correction_tensor.min().item(), correction_tensor.max().item()),
                    "grad": (cgmin, cgmax),
                    "eta": float(eta),
                    "x0_hat_interm": (x0_hat_interm.min().item(), x0_hat_interm.max().item()),
                    "x_tilde_0": (x_tilde_0.min().item(), x_tilde_0.max().item()),
                    "x_cur": (x_cur.min().item(), x_cur.max().item()),
                }
                stats_str = " ".join(f"{k}: {_fmt_stats(v)}" for k, v in stats.items())
                print(f"[Correction t={int(t.item())}, k={cstep+1}/{num_latent_corrector_steps}] {stats_str}")

        return x_cur.to(dtype=x.dtype).detach(), stats

    def calculate_reward_energy(
        x0, embC, poolC, emb_concepts, pool_concepts,
        emb_uncond=None, pool_uncond=None, rng_seed=None, num_samples_override=None,
        use_adaptive_weights=False, adaptive_zeta=None, adaptive_norm=None,
        return_components=False, x0_concepts=None,
    ):
        """Diffusion-classifier scalar reward, energy form of the PMI reward.

            R(x0) = Σ_i w_i E_{c_i}(x0) - E_C(x0),     w on the simplex

        with per-condition energy
            E_c(x0) = E_{τ,ε}[ w(τ) · || ε - ε_θ(x_τ, τ, c) ||² ],
            x_τ = √ᾱ_τ · x0 + √(1-ᾱ_τ) · ε.

        Since E_c ≈ -log p(x0|c) + const, R ≈ Σ_i w_i log[ p(x0|C) / p(x0|c_i) ], a convex
        combination of per-concept PMIs. The SAME (τ, ε) draws are shared across C and
        every c_i (common random numbers), so the difference is low-variance.
        UNet runs fp16; ε / ε_pred / the squared-sum are fp32 (no overflow).

        """
        K = emb_concepts.shape[0]
        device = x0.device
        opt2 = x0_concepts is not None
        if opt2:
            # d_i compares two CONDITIONS, so it needs a common x_τ; with per-anchor rows
            # eps_pred[0:1] - eps_pred[1:] mixes a condition change with a base-point change
            # (scratch3.tex §2.5, Remarks). Refuse rather than report a meaningless weight.
            # if use_adaptive_weights:
            #     raise NotImplementedError(
            #         "Adaptive concept weights need a common x_tau; unsupported with "
            #         "per-concept Tweedie anchors (scratch3.tex §2.5, Remarks)."
            #     )
            if x0_concepts.shape[0] != K:
                raise ValueError(
                    f"x0_concepts has {x0_concepts.shape[0]} anchors but there are {K} concepts."
                )

        # Stack [C, c_1, ..., c_K] for a single batched UNet call per (τ, ε).
        emb_stack = torch.cat([embC, emb_concepts], dim=0)
        pool_stack = torch.cat([poolC, pool_concepts], dim=0)
        use_cfg = config.energy_use_cfg and (emb_uncond is not None)
        if use_cfg:
            if opt2:
                # Each anchor needs its OWN uncond row (they sit at different x_τ), so the
                # stack is 2(K+1) rows: [uncond @ every anchor | C, c_1..c_K @ their anchor].
                emb_stack = torch.cat([emb_uncond.repeat(K + 1, 1, 1), emb_stack], dim=0)
                pool_stack = torch.cat([pool_uncond.repeat(K + 1, 1), pool_stack], dim=0)
            else:
                emb_stack = torch.cat([emb_uncond, emb_stack], dim=0)
                pool_stack = torch.cat([pool_uncond, pool_stack], dim=0)
        B = emb_stack.shape[0]

        tau_lo = int(min(config.energy_tau_min, config.energy_tau_max))
        tau_hi = int(max(config.energy_tau_min, config.energy_tau_max))
        n_samples = max(1, num_samples_override if num_samples_override is not None else config.energy_num_samples)

        # Deterministic (tau, eps) draws when rng_seed is given (used by the curl
        # diagnostic so its finite-difference field is consistent across evals).
        #! bug: fixing the rng seed to time will lead to same noise for all samples, which is not desired.
        #! also it will leak to the global rng state, which is not desired.
        if rng_seed is not None: 
            torch.manual_seed(int(rng_seed))
            if x0.is_cuda:
                torch.cuda.manual_seed_all(int(rng_seed))

        # Accumulate the per-condition energy VECTORS (and the score-difference norms) and
        # reduce once after the loop: d_i in Eq. (7) is an expectation over (τ, ε), so the
        # weights must be formed from the τ-averaged d. For the uniform path this is
        # algebraically identical to accumulating the scalar reward per sample.
        E_acc = None
        d_acc = None
        for k in range(n_samples):
            tau = stratified_tau(tau_lo, tau_hi, n_samples, k)
            tau_t = torch.tensor(tau, device=device, dtype=torch.long)
            at_tau = scheduler.alphas_cumprod[tau].to(device=device, dtype=torch.float32)
            eps = torch.randn_like(x0)  # CRN: shared across all conditions (and all anchors)
            if opt2:
                # One anchor per condition, same (tau, eps) — scratch3.tex Eq. (18).
                x0_stack = torch.cat([x0, x0_concepts], dim=0)          # [K+1]
                x_tau_rows = at_tau.sqrt() * x0_stack + (1 - at_tau).sqrt() * eps
                if use_cfg:
                    x_tau_rows = x_tau_rows.repeat(2, 1, 1, 1)          # uncond rows reuse each anchor
            else:
                x_tau = at_tau.sqrt() * x0 + (1 - at_tau).sqrt() * eps
                x_tau_rows = x_tau.repeat(B, 1, 1, 1)

            eps_pred = _predict_noise(x_tau_rows, tau_t, emb_stack, pool_stack)
            if use_cfg:
                n_cond = (K + 1) if opt2 else 1
                eps_uncond = eps_pred[:n_cond]
                eps_pred = eps_uncond + config.guidance_scale * (eps_pred[n_cond:] - eps_uncond)

            if config.energy_weight_type == "min-snr":
                snr = at_tau / (1 - at_tau)
                w = torch.clamp(snr, max=5.0) / (snr + 1e-8)
            else:
                w = x0.new_tensor(1.0)

            sq = (eps - eps_pred) ** 2                # [K+1, C, H, W] (broadcast over batch)
            E = w * sq.flatten(1).sum(dim=1)          # [K+1]
            # d_i = ω(τ)·‖ε_θ(x_τ,τ,C) - ε_θ(x_τ,τ,c_i)‖²  (Eq. 7) — free, same ε_pred rows.
            # Under Option 2 the rows sit at different anchors, so this is NOT a pure condition
            # discrepancy; it is kept as a logging-only quantity (adaptive weights are refused).
            d = w * ((eps_pred[0:1] - eps_pred[1:]) ** 2).flatten(1).sum(dim=1)  # [K]

            E_acc = E if E_acc is None else E_acc + E
            d_acc = d if d_acc is None else d_acc + d

            print(
                f"[Energy t={tau}] E_C={E[0].item():.4f} "
                f"E_concepts_mean={(E[1:].mean() if K > 0 else E[0]).item():.4f}"
            )

        E_mean = E_acc / n_samples                    # [K+1]
        d_mean = d_acc / n_samples                    # [K]
        E_C = E_mean[0]
        E_concepts = E_mean[1:]

        if K == 0:
            R = E_C - E_C
            w_i = d_mean.detach()
        elif use_adaptive_weights:
            zeta = config.energy_adaptive_zeta if adaptive_zeta is None else adaptive_zeta
            norm = config.energy_adaptive_norm if adaptive_norm is None else adaptive_norm
            # Detached weights → guidance direction is Σ_i w_i ∇_{x0}(E_{c_i} - E_C), Eq. (10).
            w_i = compute_adaptive_concept_weights(d_mean, zeta, norm_type=norm)
            R = (w_i * (E_concepts - E_C)).sum()
            w_ent = -(w_i * (w_i + 1e-12).log()).sum().item()
            d_str = "[" + ", ".join(f"{v:.3e}" for v in d_mean.detach().tolist()) + "]"
            w_str = "[" + ", ".join(f"{v:.4f}" for v in w_i.tolist()) + "]"
            print(
                f"[EnergyAdaptiveW zeta={float(zeta)} norm={norm}] "
                f"d={d_str} w={w_str} H(w)={w_ent:.4f}/{math.log(K):.4f} R={R.item():.4f}"
            )
        else:
            R = E_concepts.mean() - E_C
            w_i = torch.full_like(d_mean, 1.0 / K)

        if return_components:
            return R, {
                "w": w_i.detach(),
                "d": d_mean.detach(),
                "E_C": E_C.detach(),
                "E_concepts": E_concepts.detach(),
            }
        return R

    def _energy_particle_terms(
        x0, tau_t, at_tau, eps, emb_stack, pool_stack, use_cfg, x0_concepts=None,
    ):
        """Per-condition energies for ONE particle (a single (tau, eps) draw).

        Same math as one iteration of calculate_reward_energy's sample loop, with
        [C, c_1..c_K] (and the CFG uncond row) kept batched in a single UNet call.
        Returns (E, d): E is [K+1] = [E_C, E_c1..E_cK], d is [K] (the score-difference
        norms feeding the adaptive weights).

        With x0_concepts (Option 2) each condition is noised from its OWN Tweedie anchor
        — rows are [C @ x̂₀^C, c_i @ x̂₀^{c_i}], sharing this particle's (tau, eps) — and
        under CFG every anchor carries its own uncond row, 2(K+1) rows in total. Mirrors
        the opt2 branch of calculate_reward_energy.
        """
        opt2 = x0_concepts is not None
        if opt2:
            x0_stack = torch.cat([x0, x0_concepts], dim=0)          # [K+1]
            x_tau_rows = at_tau.sqrt() * x0_stack + (1 - at_tau).sqrt() * eps
            n_cond = x0_stack.shape[0]
            if use_cfg:
                x_tau_rows = x_tau_rows.repeat(2, 1, 1, 1)          # uncond rows reuse each anchor
        else:
            B = emb_stack.shape[0]
            x_tau = at_tau.sqrt() * x0 + (1 - at_tau).sqrt() * eps
            x_tau_rows = x_tau.repeat(B, 1, 1, 1)
            n_cond = 1

        eps_pred = _predict_noise(x_tau_rows, tau_t, emb_stack, pool_stack)
        if use_cfg:
            eps_uncond = eps_pred[:n_cond]
            eps_pred = eps_uncond + config.guidance_scale * (eps_pred[n_cond:] - eps_uncond)

        if config.energy_weight_type == "min-snr":
            snr = at_tau / (1 - at_tau)
            w = torch.clamp(snr, max=5.0) / (snr + 1e-8)
        else:
            w = x0.new_tensor(1.0)

        sq = (eps - eps_pred) ** 2
        E = w * sq.flatten(1).sum(dim=1)                                     # [K+1]
        d = w * ((eps_pred[0:1] - eps_pred[1:]) ** 2).flatten(1).sum(dim=1)  # [K]
        return E, d

    def calculate_reward_energy_grad(
        x0, embC, poolC, emb_concepts, pool_concepts,
        emb_uncond=None, pool_uncond=None, rng_seed=None, num_samples_override=None,
        use_adaptive_weights=False, adaptive_zeta=None, adaptive_norm=None,
        return_components=False, x0_concepts=None,
    ):
        """Per-particle gradient-accumulating form of calculate_reward_energy.

        Same reward, but each particle (one (tau, eps) draw) is backprop'd and freed
        before the next is drawn, instead of holding every particle's graph alive and
        backpropping once in the caller. [C, c_1..c_K] stay batched within a particle,
        so only the particle axis is traded for memory: the peak footprint is one
        particle's forward graph rather than n_samples of them.

            R = (1/n) Σ_particles Σ_i w_i (E_{c_i} - E_C)

        is linear in the per-particle terms, so accumulating ∇_{x0} per particle is
        exact — identical to one backward over the summed reward.

        CAVEAT: with use_adaptive_weights the softmin weights are formed from each
        particle's OWN d, which is what removes the need for a second pass over the
        particles. calculate_reward_energy instead applies one softmin to the
        particle-averaged d; softmin is nonlinear, so the two are not identical.

        OPTION 2 (x0_concepts given, scratch3.tex §2.5): each condition is read at its own
        Tweedie anchor, so the reward depends on K+1 separate variables and every particle
        backprops to ALL of them at once — grad_x0 is then the pair (grad_C, grad_concepts)
        with grad_C = ∂R/∂x̂₀^C (= -g_C) and grad_concepts = ∂R/∂x̂₀^{c_i} (= g_{c_i}/K),
        exactly what one grad(R, [x0, x0_concepts]) over the summed reward would give. The
        caller still owns the transport back to x_t (the Variant A VJP): only the reward
        subgraph is freed here, the anchors' base subgraph is untouched.

        Returns (grad_x0, R[, components]) — the backward already happened inside;
        grad_x0 is a tensor under Option 1, the tuple (grad_C, grad_concepts) under Option 2.
        """
        K = emb_concepts.shape[0]
        device = x0.device
        use_cfg = config.energy_use_cfg and (emb_uncond is not None)
        opt2 = x0_concepts is not None
        if opt2 and x0_concepts.shape[0] != K:
            raise ValueError(
                f"x0_concepts has {x0_concepts.shape[0]} anchors but there are {K} concepts."
            )

        # Stack [C, c_1, ..., c_K] once — one batched UNet call per particle.
        emb_stack = torch.cat([embC, emb_concepts], dim=0)
        pool_stack = torch.cat([poolC, pool_concepts], dim=0)
        if use_cfg:
            if opt2:
                # Each anchor sits at its own x_τ, so it needs its OWN uncond row: 2(K+1) rows.
                emb_stack = torch.cat([emb_uncond.repeat(K + 1, 1, 1), emb_stack], dim=0)
                pool_stack = torch.cat([pool_uncond.repeat(K + 1, 1), pool_stack], dim=0)
            else:
                emb_stack = torch.cat([emb_uncond, emb_stack], dim=0)
                pool_stack = torch.cat([pool_uncond, pool_stack], dim=0)

        tau_lo = int(min(config.energy_tau_min, config.energy_tau_max))
        tau_hi = int(max(config.energy_tau_min, config.energy_tau_max))
        n_samples = max(1, num_samples_override if num_samples_override is not None else config.energy_num_samples)

        if rng_seed is not None:
            torch.manual_seed(int(rng_seed))
            if x0.is_cuda:
                torch.cuda.manual_seed_all(int(rng_seed))

        zeta = config.energy_adaptive_zeta if adaptive_zeta is None else adaptive_zeta
        norm = config.energy_adaptive_norm if adaptive_norm is None else adaptive_norm

        grad_accum = torch.zeros_like(x0)
        grad_concepts_accum = torch.zeros_like(x0_concepts) if opt2 else None
        R_accum = x0.new_tensor(0.0)
        E_acc = x0.new_zeros(K + 1)
        d_acc = x0.new_zeros(K)
        w_acc = x0.new_zeros(K)

        for k in range(n_samples):
            tau = stratified_tau(tau_lo, tau_hi, n_samples, k)
            tau_t = torch.tensor(tau, device=device, dtype=torch.long)
            at_tau = scheduler.alphas_cumprod[tau].to(device=device, dtype=torch.float32)
            eps = torch.randn_like(x0)

            with torch.enable_grad():
                E, d = _energy_particle_terms(
                    x0, tau_t, at_tau, eps, emb_stack, pool_stack, use_cfg,
                    x0_concepts=x0_concepts,
                )
                E_C = E[0]
                E_concepts = E[1:]
                if K == 0:
                    w_i = d.detach()
                elif use_adaptive_weights:
                    # Detached weights → direction is Σ_i w_i ∇_{x0}(E_{c_i} - E_C), Eq. (10).
                    w_i = compute_adaptive_concept_weights(d.detach(), zeta, norm_type=norm)
                else:
                    w_i = torch.full_like(d, 1.0 / K)
                loss = (w_i * (E_concepts - E_C)).sum() / n_samples
                if K > 0:
                    if opt2:
                        # One backward per particle to BOTH anchor sets; the base subgraph
                        # (x̂₀^c ← x_t) is not traversed, so the caller's VJP still works.
                        g_C, g_concepts = torch.autograd.grad(
                            loss, [x0, x0_concepts], retain_graph=False,
                        )
                        grad_accum = grad_accum + g_C
                        grad_concepts_accum = grad_concepts_accum + g_concepts
                    else:
                        grad_accum = grad_accum + torch.autograd.grad(loss, x0, retain_graph=False)[0]

            R_accum = R_accum + loss.detach()
            E_acc = E_acc + E.detach()
            d_acc = d_acc + d.detach()
            w_acc = w_acc + w_i.detach()

            print(
                f"[Energy t={tau}] E_C={E[0].item():.4f} "
                f"E_concepts_mean={(E[1:].mean() if K > 0 else E[0]).item():.4f}"
                f"loss={loss.item():.4f}"
            )

        E_mean = E_acc / n_samples
        d_mean = d_acc / n_samples
        w_mean = w_acc / n_samples

        if use_adaptive_weights and K > 0:
            w_ent = -(w_mean * (w_mean + 1e-12).log()).sum().item()
            d_str = "[" + ", ".join(f"{v:.3e}" for v in d_mean.tolist()) + "]"
            w_str = "[" + ", ".join(f"{v:.4f}" for v in w_mean.tolist()) + "]"
            print(
                f"[EnergyAdaptiveW zeta={float(zeta)} norm={norm}] "
                f"d={d_str} w={w_str} H(w)={w_ent:.4f}/{math.log(K):.4f} R={R_accum.item():.4f}"
            )

        grad_out = (grad_accum, grad_concepts_accum) if opt2 else grad_accum

        if return_components:
            return grad_out, R_accum, {
                "w": w_mean, "d": d_mean,
                "E_C": E_mean[0], "E_concepts": E_mean[1:],
            }
        return grad_out, R_accum

    def compute_curl_diagnostic(
        x0, embC, poolC, emb_concepts, pool_concepts, emb_uncond, pool_uncond, t, cstep,
    ):
        """Estimate the antisymmetry of the guidance field's Jacobian.

        The field is g = ∇_{x0} R. If R is a true scalar, its Jacobian J = ∂g/∂x is
        symmetric (it is the Hessian of R) → antisymmetry ≈ 0 (conservative). Measured
        exactly via double-backward Hessian-vector products on random direction pairs:
        a = vᵀ(J - Jᵀ)u. Exact HVPs (not finite differences) avoid the catastrophic
        cancellation that would otherwise bury the signal in fp16 noise. A single
        energy sample is used (num_samples_override=1) to keep the 2nd-order graph in
        memory; non-reentrant checkpointing (already active in the caller) enables the
        double-backward. TF32 is disabled so reduced-mantissa matmuls do not inject
        fake asymmetry; the global RNG state is saved/restored.
        """
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if x0.is_cuda else None
        # NOTE: the UNet runs fp16 here, so the double-backward has a ~1e-4 noise floor.
        # Where the reward is locally near-flat, the true curvature (sym_rms) can also be
        # at that floor, making the ratio inconclusive. A decisive antisym≈0 reading needs
        # an fp32 UNet, which does not fit at 1024 — evaluate at lower resolution, or
        # compare against the score-difference baseline (run --algo_version option1).
        try:
            with torch.enable_grad():
                x = x0.detach().clone().requires_grad_(True)
                R = calculate_reward_energy(
                    x, embC, poolC, emb_concepts, pool_concepts,
                    emb_uncond=emb_uncond, pool_uncond=pool_uncond, num_samples_override=1,
                )
                g = torch.autograd.grad(R, x, create_graph=True)[0]

                n = max(1, config.curl_num_dirs)
                antisym_sq, sym_sq = 0.0, 0.0
                for _ in range(n):
                    u = torch.randn_like(x); u = u / (u.norm() + 1e-8)
                    v = torch.randn_like(x); v = v / (v.norm() + 1e-8)
                    Jtu = torch.autograd.grad(g, x, grad_outputs=u, retain_graph=True)[0]  # Jᵀu
                    Jtv = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=True)[0]  # Jᵀv
                    uJtv = (u * Jtv).sum()   # = vᵀ J u
                    vJtu = (v * Jtu).sum()   # = uᵀ J v
                    a = uJtv - vJtu          # = vᵀ(J - Jᵀ)u
                    s = 0.5 * (uJtv + vJtu)  # symmetric reference
                    antisym_sq += (a * a).item()
                    sym_sq += (s * s).item()
                antisym_rms = (antisym_sq / n) ** 0.5
                sym_rms = (sym_sq / n) ** 0.5
                ratio = antisym_rms / (sym_rms + 1e-12)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        t_val = int(t.item())
        msg = (f"[CurlDiag t={t_val} k={cstep}] antisym_rms={antisym_rms:.4e} "
               f"sym_rms={sym_rms:.4e} ratio={ratio:.4e}")
        print(msg)
        if viz_state["seed_dir"] is not None:
            out_path = os.path.join(viz_state["seed_dir"], "curl_diagnostic.txt")
            with open(out_path, "a") as f:
                f.write(msg + "\n")
        return {"antisym_rms": antisym_rms, "sym_rms": sym_rms, "ratio": ratio}

    def update_x_with_energy_grad(x, t, num_latent_corrector_steps, eta, beta, boosted_grad_ts=None, correction_mode=None):
        """`correction_mode` overrides config.correction_mode (per-phase hybrid modes)."""
        correction_mode = correction_mode or config.correction_mode
        """Scalar-energy reward guidance (see scratch2.tex).

        Replaces the score-difference composed gradient of update_x_with_dps with
        grad_x0 = ∇_{x̂₀} R, where R is the scalar diffusion-classifier reward from
        calculate_reward_energy. The gradient of a scalar is conservative by
        construction (no curl), then transported to x_t-space by the single shared
        Tweedie Jacobian. Downstream update/debug/return logic mirrors update_x_with_dps.
        """
        text_embed_uncond = text_embeds[0:1]
        text_embed_multi = text_embeds[1:2]
        text_embed_uncond_pool = text_embeds_pool[0:1]
        text_embed_multi_pool = text_embeds_pool[1:2]

        base_slice = prompt_layout["base_concept_slice"]
        attr_slice = prompt_layout["attribute_slice"]
        concept_chunks = [text_embeds[base_slice]]
        concept_pool_chunks = [text_embeds_pool[base_slice]]
        if config.use_attribute_guidance and prompt_layout["concept_items"]:
            concept_chunks.append(text_embeds[attr_slice])
            concept_pool_chunks.append(text_embeds_pool[attr_slice])
        text_embed_concepts = torch.cat([chunk for chunk in concept_chunks if chunk.shape[0] > 0], dim=0)
        text_embed_concepts_pool = torch.cat(
            [chunk for chunk in concept_pool_chunks if chunk.shape[0] > 0],
            dim=0,
        )

        if text_embed_concepts.shape[0] == 0:
            text_embed_concepts = text_embed_multi
            text_embed_concepts_pool = text_embed_multi_pool

        at, at_prev = _get_alphas(t)
        dps_t_eps = t.new_tensor(config.dps_t_eps)
        at_eps = scheduler.alphas_cumprod[dps_t_eps.cpu()]

        x_cur = x.detach().to(dtype=torch.float32)
        stats = {}
        unet.requires_grad_(False)
        # Gradient checkpointing (non-reentrant) keeps the double-backprop through
        # two chained UNet passes within memory, and enables the 2nd-order curl diag.
        ckpt_was_enabled = unet.is_gradient_checkpointing
        if not ckpt_was_enabled:
            unet.enable_gradient_checkpointing()

        with force_nonreentrant_checkpoint(), torch.autocast(device_type="cuda", enabled=False), torch.enable_grad():
            for cstep in range(num_latent_corrector_steps):
                x_cur = x_cur.detach().requires_grad_(True)

                noise_pred_base = _predict_noise(
                    torch.cat([x_cur, x_cur]),
                    t,
                    torch.cat([text_embed_uncond, text_embed_multi], dim=0),
                    torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0),
                )
                if cstep == 0:
                    x_orig = x_cur.detach().clone()
                    noise_pred_base_orig = noise_pred_base.detach().clone()

                noise_uncond = noise_pred_base[0:1]
                noise_multi = noise_pred_base[1:2]
                noise_multi_cfg = noise_uncond + config.guidance_scale * (noise_multi - noise_uncond)

                if config.x0_hat_score_source == "multi-cfg":
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_multi_cfg) / at.sqrt()
                elif config.x0_hat_score_source == "multi":
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_multi) / at.sqrt()
                else:
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_uncond) / at.sqrt()

                # ── scalar-energy reward → gradient by autodiff ───────────────────
                correction_seed = viz_state["seed"] + t
                netgrad_rng_seed = int(correction_seed.item()) if config.netgrad_crn else None 
                # With --energy_adaptive_weights the uniform 1/K is replaced by the
                # softmin weights of scratch4.tex Eq. (9); off by default → Eq. (6).
                # NOTE: the field is conservative only in the uniform case — the adaptive
                # weights are detached, so the Σ_i r_i ∇w_i term is dropped (see the
                # CAVEAT in calculate_reward_energy).
                grad_x0, R, reward_comps = calculate_reward_energy_grad(
                    x0_hat,
                    text_embed_multi, text_embed_multi_pool,
                    text_embed_concepts, text_embed_concepts_pool,
                    emb_uncond=text_embed_uncond, pool_uncond=text_embed_uncond_pool,
                    rng_seed=netgrad_rng_seed,
                    use_adaptive_weights=config.energy_adaptive_weights,
                    return_components=True,
                )
                # ∇_{x̂₀} R — field in x̂₀-space (drop-in for composed_score_x0), accumulated
                # per-particle inside calculate_reward_energy_grad (see docstring there).
                composed_score_x0 = grad_x0   # alias for debug-save blocks
                score_concept_x0 = grad_x0    # alias for correct-tweedie save block

                if correction_mode == "correct-tweedie":
                    # Algorithm B (x̂₀-space, no Jacobian): correct the Tweedie mean
                    # directly with the x̂₀-space gradient; skip the base-pass VJP.
                    grad = (beta * grad_x0).detach()
                else:
                    # Algorithm A (x_t-space): transport to x_t via the shared Tweedie Jacobian.
                    grad = torch.autograd.grad(
                        outputs=x0_hat,
                        inputs=x_cur,
                        grad_outputs=grad_x0,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )[0] * beta

                # Optional curl / conservativity diagnostic (debug only).
                if (config.run_curl_diagnostic and config.save_intermediates
                        and viz_state["seed_dir"] is not None
                        and cstep in [0, num_latent_corrector_steps - 1]):
                    compute_curl_diagnostic(
                        x0_hat.detach(),
                        text_embed_multi, text_embed_multi_pool,
                        text_embed_concepts, text_embed_concepts_pool,
                        text_embed_uncond, text_embed_uncond_pool,
                        t, cstep,
                    )

                if correction_mode == "reverse-sde":
                    cgmin, cgmax = grad.min().item(), grad.max().item()

                    x_interm = x_cur.detach()  # stay at the same timestep

                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    # x0_hat_interm = _compute_x0_hat_interm(x_interm.detach(), at, noise_uncond, noise_multi, noise_multi_cfg)
                    x_tilde_0 = (1 - eta / num_latent_corrector_steps) * x0_hat_interm + (eta / num_latent_corrector_steps) * grad.detach()

                    grad = grad.detach() / (grad.abs().max() + 1e-3) * x0_hat_interm.abs().max()  # force normalize to avoid magnitude issues
                    correction_tensor = grad  # (energy path) define so stats/debug stay consistent
                    x_cur = (x_interm + (float(eta) * at.sqrt()/ num_latent_corrector_steps) * grad).detach()
                    x_cur = x_cur / (x_cur.abs().max() + 1e-3) * x_orig.abs().max()  # force normalize to avoid magnitude issues

                    # final outer update at the end of the loop
                    if cstep == num_latent_corrector_steps - 1:
                        net_grad = (x_cur - x_orig).detach()
                        x_cur = _get_x_t_prev(
                            x_t=x_orig,
                            noise_pred_base=noise_pred_base_orig,
                            net_grad=net_grad,
                            t=t,
                            at=at,
                            at_prev=at_prev,
                            eta=eta,
                            guidance_scale=config.guidance_scale,
                            update_step_type=config.update_step_type,
                            x0_interm_noise_type=config.x0_final_interm_noise_type, #should be final interm noise
                            use_cfgpp=config.use_cfgpp,
                        )

                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:
                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")

                        for _name, _tensor in [
                            ("x0_hat", x0_hat),
                            ("composed_score_x0", composed_score_x0),
                            ("grad", grad),
                            ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_cur", x_cur),
                            ("x_tilde_0", x_tilde_0),
                        ]:
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_raw.png")
                            viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val} k={cstep}")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_decoded.png")
                            decode_tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(decode_tensor, out_path)
                elif correction_mode == "correct-tweedie":
                    # Algorithm B: correct x̂₀ directly in x̂₀-space (no Jacobian), then renoise.
                    # `grad` here is the x̂₀-space correction β·∇_{x̂₀}R (set above).
                    cgmin, cgmax = grad.min().item(), grad.max().item()
                    # grad = grad.clamp(-5, 5) #! check approproiate value

                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    snr = (1 - at).sqrt() / at.sqrt()  # can be used to schedule eta # this snr formula is wrong
                    # x̂₀ ← x̂₀ + η·β·∇_{x̂₀}R  (additive correction, not a convex combination)
                    # grad = grad.detach() / (grad.norm() + 1e-4) * x0_hat_interm.norm()  #! force normalize to the strength of the Tweedie mean to avoid magnitude issues
                    grad = grad.detach() / (grad.abs().max() + 1e-3) * x0_hat_interm.abs().max()
                    # (x0_factor, grad_factor) per position within boosted_grad_ts; last entry
                    # is reused if boosted_grad_ts is longer than this schedule.
                    boosted_grad_factors = [(0.2, 0.8), (0.3, 0.7)]
                    if boosted_grad_ts is not None and t in boosted_grad_ts:
                        b_idx = int((boosted_grad_ts == t).nonzero()[0].item())
                        x0_factor, grad_factor = boosted_grad_factors[min(b_idx, len(boosted_grad_factors) - 1)]
                    else:
                        x0_factor = 1.0
                        grad_factor = (float(eta)/num_latent_corrector_steps) # eta_factor is 1.0 for correct-tweedie
                    print(f"[DEBUG] t={t.item()} x0_factor={x0_factor} grad_factor={grad_factor}")
                    x_tilde_0 = x0_factor * x0_hat_interm + grad_factor * grad.detach()
                    
                    #! Force-normalize to avoid magnitude drift
                    x_tilde_0 = x_tilde_0 / (x_tilde_0.abs().max() + 1e-3) * x0_hat_interm.abs().max()
                    correction_tensor = _x0_to_noise(x_cur.detach(), x_tilde_0, at)

                    # debug
                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:

                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")

                        for _name, _tensor in [
                            # ("x0_hat", x0_hat.detach()),
                            # ("score_concept_x0", score_concept_x0.detach()),
                            # ("composed_score_x0", composed_score_x0),
                            ("grad", grad),
                            # ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_tilde_0", x_tilde_0),
                        ]:
                            if torch.isnan(_tensor).any() or torch.isinf(_tensor).any():
                                print(f"[WARN] {_name} has NaN/Inf")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_raw.png")
                            viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val} k={cstep}")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_decoded.png")
                            decode_tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(decode_tensor, out_path)

                    # forward_noise = noise_uncond if config.use_cfgpp else noise_multi_cfg  #! old setting
                    forward_noise = noise_interm #! New setting
                    x_cur = _update_x_cur_from_x_tilde(
                        x_cur.detach(), cstep, num_latent_corrector_steps, config.use_cfgpp, config.psi,
                        at, at_prev, x_tilde_0, noise_uncond, noise_multi,
                        forward_noise=forward_noise 
                    )
                    x_cur = x_cur.detach()
                else:
                    x_intermediate = scheduler.step(noise_multi_cfg, t, x_cur.detach(), return_dict=False)[0]
                    correction_tensor = grad
                    x_cur = (x_intermediate + float(eta) * grad).detach()

                stats = {
                    "mode": correction_mode,
                    "correction_tensor": (correction_tensor.min().item(), correction_tensor.max().item()),
                    "grad": (cgmin, cgmax),
                    "eta": float(eta),
                    "x0_hat_interm": (x0_hat_interm.min().item(), x0_hat_interm.max().item()),
                    "x_tilde_0": (x_tilde_0.min().item(), x_tilde_0.max().item()),
                    "x_cur": (x_cur.min().item(), x_cur.max().item()),
                    "reward": float(R.detach().item()),
                }
                if config.energy_adaptive_weights:
                    w_i = reward_comps["w"]
                    stats["w"] = (w_i.min().item(), w_i.max().item())
                    stats["w_entropy"] = float(-(w_i * (w_i + 1e-12).log()).sum().item())
                stats_str = " ".join(f"{k}: {_fmt_stats(v)}" for k, v in stats.items())
                print(f"[EnergyCorrection t={int(t.item())}, k={cstep+1}/{num_latent_corrector_steps}] {stats_str}")

        if not ckpt_was_enabled:
            unet.disable_gradient_checkpointing()

        return x_cur.to(dtype=x.dtype).detach(), stats

    def update_x_with_energy_grad_opt2(x, t, num_latent_corrector_steps, eta, beta, boosted_grad_ts=None, correction_mode=None):
        """`correction_mode` overrides config.correction_mode (per-phase hybrid modes)."""
        correction_mode = correction_mode or config.correction_mode
        """Option 2 of the scalar-energy reward: concept-wise Tweedie anchors (scratch3.tex §2.5).

        update_x_with_energy_grad reads all K+1 energies at ONE point, the full-prompt Tweedie
        mean. Here each energy is read at its own anchor (Eq. 16),

            x̂₀^c = 𝒯_t(x_t, ε^cfg_c(x_t, t)),      c ∈ {C, c_1, ..., c_K},

        because the tightest posterior mean for the c-term is the one taken UNDER c. That gives

            R⁽²⁾ = (1/K) Σ_i E_{c_i}(x̂₀^{c_i}) - E_C(x̂₀^C)          (Eq. 17)

        with g_c = ∇_u E_c(u)|_{u = x̂₀^c}. BOTH correction modes now use Variant A (Eq. 20),

            ∇_{x_t}R⁽²⁾ = (1/K) Σ_i J(x̂₀^{c_i})^T g_{c_i} - J(x̂₀^C)^T g_C

        i.e. the Jacobians are always kept. The Jacobian-free Variant B (Eq. 21) is NOT reachable
        any more: its G⁽²⁾ sums gradients w.r.t. K+1 different variables, so it is the gradient of
        no scalar. The modes now differ only in where that one gradient is applied:
        """
        # Fail fast, before any sampling: d_i needs a common x_τ (see calculate_reward_energy).
        # if config.energy_adaptive_weights:
        #     raise NotImplementedError(
        #         "--energy_adaptive_weights is unsupported with algo_version=energy-opt2: the "
        #         "discrepancies d_i need a common x_tau (scratch3.tex §2.5, Remarks)."
        #     )
        if config.x0_hat_score_source not in ("multi-cfg", "multi"):
            print(
                "[WARN] energy-opt2 with --x0_hat_score_source="
                f"{config.x0_hat_score_source}: every anchor collapses to the same uncond "
                "Tweedie mean, so Option 2 degenerates to Option 1 (scratch3.tex Eq. 26, last "
                "line) — this run is equivalent to --algo_version energy, at extra cost."
            )

        text_embed_uncond = text_embeds[0:1]
        text_embed_multi = text_embeds[1:2]
        text_embed_uncond_pool = text_embeds_pool[0:1]
        text_embed_multi_pool = text_embeds_pool[1:2]

        base_slice = prompt_layout["base_concept_slice"]
        attr_slice = prompt_layout["attribute_slice"]
        concept_chunks = [text_embeds[base_slice]]
        concept_pool_chunks = [text_embeds_pool[base_slice]]
        if config.use_attribute_guidance and prompt_layout["concept_items"]:
            concept_chunks.append(text_embeds[attr_slice])
            concept_pool_chunks.append(text_embeds_pool[attr_slice])
        text_embed_concepts = torch.cat([chunk for chunk in concept_chunks if chunk.shape[0] > 0], dim=0)
        text_embed_concepts_pool = torch.cat(
            [chunk for chunk in concept_pool_chunks if chunk.shape[0] > 0],
            dim=0,
        )

        if text_embed_concepts.shape[0] == 0:
            text_embed_concepts = text_embed_multi
            text_embed_concepts_pool = text_embed_multi_pool

        K = text_embed_concepts.shape[0]
        # Base stack [∅, C, c_1, ..., c_K] — K+2 rows, all at the same x_t.
        text_embed_base = torch.cat([text_embed_uncond, text_embed_multi, text_embed_concepts], dim=0)
        text_embed_base_pool = torch.cat(
            [text_embed_uncond_pool, text_embed_multi_pool, text_embed_concepts_pool], dim=0
        )

        at, at_prev = _get_alphas(t)
        dps_t_eps = t.new_tensor(config.dps_t_eps)
        at_eps = scheduler.alphas_cumprod[dps_t_eps.cpu()]

        x_cur = x.detach().to(dtype=torch.float32)
        stats = {}
        unet.requires_grad_(False)
        ckpt_was_enabled = unet.is_gradient_checkpointing
        if not ckpt_was_enabled:
            unet.enable_gradient_checkpointing()

        with force_nonreentrant_checkpoint(), torch.autocast(device_type="cuda", enabled=False), torch.enable_grad():
            for cstep in range(num_latent_corrector_steps):
                x_cur = x_cur.detach().requires_grad_(True)

                noise_pred_base = _predict_noise(
                    x_cur.repeat(K + 2, 1, 1, 1),
                    t,
                    text_embed_base,
                    text_embed_base_pool,
                )
                if cstep == 0:
                    x_orig = x_cur.detach().clone()
                    noise_pred_base_orig = noise_pred_base.detach().clone()
                    

                noise_uncond = noise_pred_base[0:1]
                noise_multi = noise_pred_base[1:2]
                noise_concepts = noise_pred_base[2:]                       # [K]
                noise_multi_cfg = noise_uncond + config.guidance_scale * (noise_multi - noise_uncond)
                noise_concepts_cfg = noise_uncond + config.guidance_scale * (noise_concepts - noise_uncond)

                # ── the K+1 Tweedie anchors, Eq. (16) ─────────────────────────────
                if config.x0_hat_score_source == "multi-cfg":
                    x0_hat_C = (x_cur - (1 - at).sqrt() * noise_multi_cfg) / at.sqrt()
                    x0_hat_concepts = (x_cur - (1 - at).sqrt() * noise_concepts_cfg) / at.sqrt()
                elif config.x0_hat_score_source == "multi":
                    x0_hat_C = (x_cur - (1 - at).sqrt() * noise_multi) / at.sqrt()
                    x0_hat_concepts = (x_cur - (1 - at).sqrt() * noise_concepts) / at.sqrt()
                else:
                    # Degenerate: every anchor coincides → opt2 reduces to opt1 (warned above).
                    x0_hat_C = (x_cur - (1 - at).sqrt() * noise_uncond) / at.sqrt()
                    # Build as a SIBLING of x0_hat_C off x_cur, not as a view of it: if
                    # x0_hat_concepts were derived from x0_hat_C, backprop would accumulate the
                    # concept terms into grad_C too and the G^(2) below would double-count them.
                    x0_hat_concepts = (x_cur.repeat(K, 1, 1, 1) - (1 - at).sqrt() * noise_uncond) / at.sqrt()

                # ── scalar-energy reward at matched anchors → gradient by autodiff ─
                correction_seed = viz_state["seed"] + t
                netgrad_rng_seed = int(correction_seed.item()) if config.netgrad_crn else None 
                # Partials at each anchor: ∂R/∂x̂₀^C = -g_C, ∂R/∂x̂₀^{c_i} = g_{c_i}/K —
                # accumulated ONE PARTICLE AT A TIME inside calculate_reward_energy_grad, so
                # only a single particle's forward graph is ever resident instead of all
                # n_samples of them (see the docstring there). Reward and gradient are
                # unchanged in the uniform-weight case; with --energy_adaptive_weights the
                # weights come from each particle's own d (the CAVEAT there).
                (grad_C, grad_concepts), R, reward_comps = calculate_reward_energy_grad(
                    x0_hat_C,
                    text_embed_multi, text_embed_multi_pool,
                    text_embed_concepts, text_embed_concepts_pool,
                    emb_uncond=text_embed_uncond, pool_uncond=text_embed_uncond_pool,
                    rng_seed=netgrad_rng_seed,
                    use_adaptive_weights=config.energy_adaptive_weights,
                    x0_concepts=x0_hat_concepts,
                    return_components=True,
                )
                # Jacobian-free G^(2) (Eq. 21). No longer used for the update — kept for the viz
                # blocks, as a reference for what the dropped-Jacobian form would have given.
                grad_x0 = grad_C + grad_concepts.sum(dim=0, keepdim=True)
                composed_score_x0 = grad_x0
                score_concept_x0 = grad_concepts.sum(dim=0, keepdim=True)

                # Variant A (Eq. 20), used by BOTH modes: one VJP of the K+1 partials back to x_t
                # gives ∇_{x_t}R^(2) = (1/K)Σ_i J(x̂₀^{c_i})^T g_{c_i} - J(x̂₀^C)^T g_C directly.
                # Touches the base subgraph only — the reward subgraph was freed by the grad()
                # above — so it is cheaper than grad(R, x_cur), which would redo the reward passes.
                grad = torch.autograd.grad(
                    outputs=[x0_hat_C, x0_hat_concepts],
                    inputs=x_cur,
                    grad_outputs=[grad_C, grad_concepts],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0] * beta

                if correction_mode == "reverse-sde":
                    cgmin, cgmax = grad.min().item(), grad.max().item()

                    x_interm = x_cur.detach()  # stay at the same timestep

                    # x0_hat_interm = _compute_x0_hat_interm(x_interm.detach(), at, noise_uncond, noise_multi, noise_multi_cfg)
                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    # x_tilde_0 = (1 - eta / num_latent_corrector_steps) * x0_hat_interm + (eta / num_latent_corrector_steps) * grad.detach() #!! eta is very low at high t; so this is mostly grad
                    x_tilde_0 = x0_hat_interm + eta / (num_latent_corrector_steps) * grad.detach() # divide by at_prev because it was pre-multiplied inside denoise_step() for x_t correction. 

                    # Erases any constant prefactor on grad, so beta is inert here (eta is the
                    # live knob). Load-bearing: it is also what absorbs the x_t-vs-x̂₀ scale gap.
                    grad = grad.detach() / (grad.abs().max() + 1e-3) * x0_hat_interm.abs().max()
                    correction_tensor = grad  # (energy path) define so stats/debug stay consistent
                    x_cur = (x_interm + (float(eta) * at.sqrt() / num_latent_corrector_steps) * grad).detach()
                    x_cur = (x_cur / (x_cur.abs().max() + 1e-3) * x_orig.abs().max()).detach()  #! force-normalize to avoid magnitude drift

                    # final outer update at the end of the loop: one step along the NET gradient
                    # accumulated by the inner steps (inner uses eta/M, this outer step uses eta).
                    if cstep == num_latent_corrector_steps - 1:
                        net_grad = (x_cur - x_orig).detach()
                        x_cur = _get_x_t_prev(
                            x_t=x_orig,
                            noise_pred_base=noise_pred_base_orig,  # reads rows [0:1]/[1:2] only
                            net_grad=net_grad,
                            t=t,
                            at=at,
                            at_prev=at_prev,
                            eta=eta,
                            guidance_scale=config.guidance_scale,
                            update_step_type=config.update_step_type,
                            x0_interm_noise_type=config.x0_final_interm_noise_type, # SEE
                            use_cfgpp=config.use_cfgpp,
                        )

                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:
                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")

                        for _name, _tensor in [
                            # ("x0_hat_C", x0_hat_C.detach()),
                            ("x0_hat_concepts_mean", x0_hat_concepts.detach().mean(dim=0, keepdim=True)),
                            ("grad_x0", grad_x0),
                            # ("composed_score_x0", composed_score_x0),
                            ("grad", grad),
                            # ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_cur", x_cur),
                            ("x_tilde_0", x_tilde_0),
                        ]:
                            if torch.isnan(_tensor).any() or torch.isinf(_tensor).any():
                                print(f"[WARN] {_name} has NaN/Inf")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_raw.png")
                            viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val} k={cstep}")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_decoded.png")
                            decode_tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(decode_tensor, out_path)
                elif correction_mode == "correct-tweedie":
                    #! This branch's update is not mathematically consistent -- we add grad w.r.t x_t to x0_hat. 
                    # Apply the SAME Eq. 20 gradient, but in x̂₀-space: add it to x̂₀^C, then
                    # re-noise. `grad` is x_t-space; the normalize below absorbs the 1/√ᾱ_t gap.
                    cgmin, cgmax = grad.min().item(), grad.max().item()

                    # x0_hat_interm = _compute_x0_hat_interm(x_cur.detach(), at, noise_uncond, noise_multi, noise_multi_cfg)
                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    snr = (1 - at).sqrt() / at.sqrt()  # can be used to schedule eta
                    # As in reverse-sde: erases the prefactor, so beta is inert here too.
                    grad = grad.detach() / (grad.abs().max() + 1e-3) * x0_hat_interm.abs().max()
                    # (x0_factor, grad_factor) per position within boosted_grad_ts; last entry
                    # is reused if boosted_grad_ts is longer than this schedule.
                    boosted_grad_factors = [(0.3, 0.7), (0.4, 0.6)]
                    if boosted_grad_ts is not None and t in boosted_grad_ts:
                        b_idx = int((boosted_grad_ts == t).nonzero()[0].item())
                        x0_factor, grad_factor = boosted_grad_factors[min(b_idx, len(boosted_grad_factors) - 1)]
                    else:
                        x0_factor = 1.0
                        grad_factor = (float(eta)/ num_latent_corrector_steps) # if no net_grad step, eta is the right factor to use.
                    print(f"[DEBUG] t={t.item()} x0_factor={x0_factor} grad_factor={grad_factor}")
                    x_tilde_0 = x0_factor * x0_hat_interm + grad_factor * grad.detach()

                    correction_tensor = _x0_to_noise(x_cur.detach(), x_tilde_0, at)

                    # debug
                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:

                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")

                        for _name, _tensor in [
                            # ("x0_hat_C", x0_hat_C.detach()),
                            ("x0_hat_concepts_mean", x0_hat_concepts.detach().mean(dim=0, keepdim=True)),
                            ("score_concept_x0", score_concept_x0.detach()),
                            # ("composed_score_x0", composed_score_x0),
                            ("grad", grad),
                            # ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_tilde_0", x_tilde_0),
                        ]:
                            if torch.isnan(_tensor).any() or torch.isinf(_tensor).any():
                                print(f"[WARN] {_name} has NaN/Inf")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_raw.png")
                            viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val} k={cstep}")
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_decoded.png")
                            decode_tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(decode_tensor, out_path)

                    # final_step_cond_noise = noise_uncond if config.use_cfgpp else noise_multi_cfg
                    # forward_noise = noise_uncond if config.use_cfgpp else noise_multi_cfg #! old setting
                    forward_noise = noise_interm #! New setting
                    x_cur = _update_x_cur_from_x_tilde(
                        x_cur.detach(), cstep, num_latent_corrector_steps, config.use_cfgpp, config.psi,
                        at, at_prev, x_tilde_0, noise_uncond, noise_multi,
                        forward_noise=forward_noise,
                    )
                    x_cur = x_cur.detach()
                else:
                    raise ValueError(f"Unknown correction_mode: {correction_mode}")

                # ‖x̂₀^C - x̂₀^{c_i}‖ = √(1-ᾱ_t)/√ᾱ_t · ‖Δε‖ (scratch3.tex Eq. 27): controls how
                # much CRN still buys us, and how far Option 2 sits from Option 1. SHRINKS as
                # t → 0 (measured 29 → 0.17 over t=951→1), so this is an early-t intervention.
                anchor_gap = (x0_hat_concepts.detach() - x0_hat_C.detach()).flatten(1).norm(dim=1)
                stats = {
                    "mode": correction_mode,
                    "correction_tensor": (correction_tensor.min().item(), correction_tensor.max().item()),
                    "grad": (cgmin, cgmax),
                    "eta": float(eta),
                    "x0_hat_interm": (x0_hat_interm.min().item(), x0_hat_interm.max().item()),
                    "x_tilde_0": (x_tilde_0.min().item(), x_tilde_0.max().item()),
                    "x_cur": (x_cur.min().item(), x_cur.max().item()),
                    "anchor_gap": (anchor_gap.min().item(), anchor_gap.max().item()),
                    "g_C": (grad_C.min().item(), grad_C.max().item()),
                    "reward": float(R.detach().item())
                }
                stats_str = " ".join(f"{k}: {_fmt_stats(v)}" for k, v in stats.items())
                print(f"[EnergyOpt2Correction t={int(t.item())}, k={cstep+1}/{num_latent_corrector_steps}] {stats_str}")

        if not ckpt_was_enabled:
            unet.disable_gradient_checkpointing()

        return x_cur.to(dtype=x.dtype).detach(), stats

    def update_x_with_energy_eucl_netgrad(x, t, num_latent_corrector_steps, eta, beta, correction_mode=None):
        """`correction_mode` overrides config.correction_mode (per-phase hybrid modes)."""
        correction_mode = correction_mode or config.correction_mode
        """Euclidean net-gradient ascent in x̂₀-space (see scratch3.tex §2).

        Unlike update_x_with_energy_grad/correct-tweedie — which re-noises back to t and
        re-runs the base UNet pass at every inner step — the base pass here runs ONCE and
        the whole corrector loop stays in x̂₀-space:

            x_t → x̂₀⁽¹⁾ --∇R--> x̃₀⁽²⁾ --∇R--> ... → x̃₀⁽ᴹ⁾

        with, for k = 1..M (M = num_latent_corrector_steps gradient evaluations),
            ĝ⁽ᵏ⁾ = β·∇_{u⁽ᵏ⁾}R / (‖·‖+δ) · ‖x̂₀⁽¹⁾‖      (norm-match to the ANCHOR)
            u⁽ᵏ⁺¹⁾ = u⁽ᵏ⁾ + (γ/M)·ĝ⁽ᵏ⁾
        The accumulated displacement is the net gradient, and one outer step is taken along it:
            Δ = u⁽ᴹ⁺¹⁾ - x̂₀⁽¹⁾ ,   x̃₀ = x̂₀⁽¹⁾ + η·Δ
        Finally one DDIM step reuses the ORIGINAL ε that produced x̂₀⁽¹⁾, so (ε, x̂₀) never
        become mismatched: x_{t-1} = √ᾱ_{t-1}·x̃₀ + √(1-ᾱ_{t-1})·ε.

        Consequences vs. correct-tweedie: no re-noising (successive gradients live on one
        landscape → R should rise monotonically across inner steps), ‖Δ‖ ≤ γ‖x̂₀⁽¹⁾‖ regardless
        of M, and one base UNet pass per timestep instead of M.

        Knobs: γ = config.netgrad_gamma is the inner step size; η scales the ACCUMULATED
        displacement (η=1 → plain accumulated ascent, x̃₀ = x̃₀⁽ᴹ⁾). The anchor ε is selected
        by --x0_interm_noise_type; --x0_hat_score_source is unused on this path (a single
        anchor serves both as the gradient base point and as the corrected Tweedie mean).
        """
        if num_latent_corrector_steps < 1:
            raise ValueError("energy-netgrad needs num_latent_corrector_steps >= 1 (M gradient evaluations).")

        text_embed_uncond = text_embeds[0:1]
        text_embed_multi = text_embeds[1:2]
        text_embed_uncond_pool = text_embeds_pool[0:1]
        text_embed_multi_pool = text_embeds_pool[1:2]

        base_slice = prompt_layout["base_concept_slice"]
        attr_slice = prompt_layout["attribute_slice"]
        concept_chunks = [text_embeds[base_slice]]
        concept_pool_chunks = [text_embeds_pool[base_slice]]
        if config.use_attribute_guidance and prompt_layout["concept_items"]:
            concept_chunks.append(text_embeds[attr_slice])
            concept_pool_chunks.append(text_embeds_pool[attr_slice])
        text_embed_concepts = torch.cat([chunk for chunk in concept_chunks if chunk.shape[0] > 0], dim=0)
        text_embed_concepts_pool = torch.cat(
            [chunk for chunk in concept_pool_chunks if chunk.shape[0] > 0],
            dim=0,
        )

        if text_embed_concepts.shape[0] == 0:
            text_embed_concepts = text_embed_multi
            text_embed_concepts_pool = text_embed_multi_pool

        at, at_prev = _get_alphas(t)

        x_cur = x.detach().to(dtype=torch.float32)
        stats = {}
        unet.requires_grad_(False)
        # Gradient checkpointing (non-reentrant) keeps the backprop through the reward's
        # UNet passes within memory, and enables the 2nd-order curl diag.
        ckpt_was_enabled = unet.is_gradient_checkpointing
        if not ckpt_was_enabled:
            unet.enable_gradient_checkpointing()

        with force_nonreentrant_checkpoint(), torch.autocast(device_type="cuda", enabled=False), torch.enable_grad():
            # ── base pass: ONCE, and without grad (no Jacobian is transported here) ──
            with torch.no_grad():
                noise_pred_base = _predict_noise(
                    torch.cat([x_cur, x_cur]),
                    t,
                    torch.cat([text_embed_uncond, text_embed_multi], dim=0),
                    torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0),
                )
            noise_uncond = noise_pred_base[0:1]
            noise_multi = noise_pred_base[1:2]
            noise_multi_cfg = noise_uncond + config.guidance_scale * (noise_multi - noise_uncond)

            # ── frozen anchor x̂₀⁽¹⁾ and the ε that produced it (reused by the DDIM step) ──
            if config.x0_interm_noise_type == "multi-cfg":
                eps_anchor = noise_multi_cfg.detach()
            elif config.x0_interm_noise_type == "multi":
                eps_anchor = noise_multi.detach()
            else:
                eps_anchor = noise_uncond.detach()
            x0_hat_anchor = _compute_x0_from_xt(x_cur, at, eps_anchor)
            anchor_norm = x0_hat_anchor.abs().norm()

            # Optional curl / conservativity diagnostic on the anchor (debug only).
            if (config.run_curl_diagnostic and config.save_intermediates
                    and viz_state["seed_dir"] is not None):
                compute_curl_diagnostic(
                    x0_hat_anchor.detach(),
                    text_embed_multi, text_embed_multi_pool,
                    text_embed_concepts, text_embed_concepts_pool,
                    text_embed_uncond, text_embed_uncond_pool,
                    t, 0,
                )

            # ── inner Euclidean ascent loop: no re-noising, no new noise prediction ──
            # With --netgrad_crn the reward's (tau, eps) draws are pinned to one seed for the
            # whole inner loop, so every g^(k) is a gradient of the SAME fixed landscape (the
            # ascent property the re-noising loop could not give). The global RNG state is
            # saved/restored around the loop so downstream sampling noise is unaffected.
            correction_seed = viz_state["seed"] + t
            netgrad_rng_seed = int(correction_seed.item()) if config.netgrad_crn else None 
            if config.netgrad_crn:
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state_all() if x_cur.is_cuda else None

            u = x0_hat_anchor.detach().clone()
            for cstep in range(num_latent_corrector_steps):
                u = u.detach().requires_grad_(True)

                R = calculate_reward_energy(
                    u,
                    text_embed_multi, text_embed_multi_pool,
                    text_embed_concepts, text_embed_concepts_pool,
                    emb_uncond=text_embed_uncond, pool_uncond=text_embed_uncond_pool,
                    rng_seed=netgrad_rng_seed,
                )
                grad = torch.autograd.grad(R, u, retain_graph=False)[0] * beta
                cgmin, cgmax = grad.min().item(), grad.max().item()
                # Norm-match to the ANCHOR (not to u): fixes the scale of every inner step,
                # so ‖Δ‖ ≤ γ·‖x̂₀⁽¹⁾‖ no matter how large M is.
                grad = grad.detach() / (grad.abs().norm() + 1e-3) * anchor_norm
                u = (u.detach() + (float(config.netgrad_gamma) / num_latent_corrector_steps) * grad).detach()

                if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:
                    t_val = int(t.item())
                    tcorr = viz_state["correction_count"]
                    inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")

                    for _name, _tensor in [
                        ("x0_hat_anchor", x0_hat_anchor.detach()),
                        ("grad", grad),
                        ("u", u),
                    ]:
                        if torch.isnan(_tensor).any() or torch.isinf(_tensor).any():
                            print(f"[WARN] {_name} has NaN/Inf")
                        out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_raw.png")
                        viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val} k={cstep}")
                        out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_decoded.png")
                        decode_tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                        decode_latent_for_viz(decode_tensor, out_path)

                # With no re-noising every gradient lives on one landscape, so R should
                # increase monotonically across the inner loop — the built-in sanity check.
                print(
                    f"[EuclNetGrad t={int(t.item())}, k={cstep+1}/{num_latent_corrector_steps}] "
                    f"reward: {float(R.detach().item()):.4f} grad: {_fmt_stats((cgmin, cgmax))} "
                    f"u: {_fmt_stats((u.min().item(), u.max().item()))}"
                )

            if config.netgrad_crn:
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)

            # ── net gradient Δ and the single outer step along it ──
            net_grad = (u - x0_hat_anchor).detach()
            # Normalize the net gradient to have the same magnitude as the anchor
            net_grad = net_grad / (net_grad.abs().max() + 1e-3) * x0_hat_anchor.abs().max()
            x_tilde_0 = (x0_hat_anchor + float(eta) * net_grad).detach() #Todo: need to calibrate eta: snr dependent. 
            # ε_eff: the whole correction folded back into a shifted noise prediction (logging).
            correction_tensor = _x0_to_noise(x_cur.detach(), x_tilde_0, at)

            # ── one DDIM step with the ORIGINAL ε (final-step branch of the helper) ──
            x_cur = _update_x_cur_from_x_tilde(
                x_cur.detach(), num_latent_corrector_steps - 1, num_latent_corrector_steps,
                config.use_cfgpp, config.psi,
                at, at_prev, x_tilde_0, noise_uncond, noise_multi,
                forward_noise=eps_anchor,
            ).detach()

            if config.save_intermediates and viz_state["seed_dir"] is not None:
                t_val = int(t.item())
                tcorr = viz_state["correction_count"]
                inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")

                for _name, _tensor in [
                    ("net_grad", net_grad),
                    ("x_tilde_0", x_tilde_0),
                    ("correction_tensor", correction_tensor),
                ]:
                    if torch.isnan(_tensor).any() or torch.isinf(_tensor).any():
                        print(f"[WARN] {_name} has NaN/Inf")
                    out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_{_name}_raw.png")
                    viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val}")
                    out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_{_name}_decoded.png")
                    decode_tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                    decode_latent_for_viz(decode_tensor, out_path)

            stats = {
                "mode": "eucl-netgrad",
                "correction_tensor": (correction_tensor.min().item(), correction_tensor.max().item()),
                "grad": (cgmin, cgmax),
                "eta": float(eta),
                "gamma": float(config.netgrad_gamma),
                "x0_hat_interm": (x0_hat_anchor.min().item(), x0_hat_anchor.max().item()),
                "net_grad": (net_grad.min().item(), net_grad.max().item()),
                "x_tilde_0": (x_tilde_0.min().item(), x_tilde_0.max().item()),
                "x_cur": (x_cur.min().item(), x_cur.max().item()),
                "reward": float(R.detach().item()),
            }
            stats_str = " ".join(f"{k}: {_fmt_stats(v)}" for k, v in stats.items())
            print(f"[EuclNetGrad t={int(t.item())}] {stats_str}")
            # Bounded-correction check: ‖Δ‖ ≤ γ·‖x̂₀⁽¹⁾‖ regardless of M.
            print(
                f"[EuclNetGrad t={int(t.item())}] |net_grad|={net_grad.norm().item():.4f} "
                f"gamma*|x0_hat_anchor|={float(config.netgrad_gamma) * anchor_norm.item():.4f}"
            )

        if not ckpt_was_enabled:
            unet.disable_gradient_checkpointing()

        return x_cur.to(dtype=x.dtype).detach(), stats

    def update_x_with_dps_option2(x, t, num_latent_corrector_steps, eta, beta, correction_mode=None):
        """`correction_mode` overrides config.correction_mode (per-phase hybrid modes)."""
        correction_mode = correction_mode or config.correction_mode
        """Option 2: per-concept Jacobians.

        grad = (β+1) · J^T_{x_t}(x̂^C_0) · s(x̂^C_0, t=0, C)
                      - β · Σ_i  J^T_{x_t}(x̂^{c_i}_0) · s(x̂^{c_i}_0, t=0, c_i)

        Each x̂_0 is estimated with its own conditioning, so each Jacobian
        captures how that specific x̂_0 moves with x_t.
        """
        text_embed_uncond = text_embeds[0:1]
        text_embed_multi = text_embeds[1:2]
        text_embed_uncond_pool = text_embeds_pool[0:1]
        text_embed_multi_pool = text_embeds_pool[1:2]

        base_slice = prompt_layout["base_concept_slice"]
        concept_chunks = [text_embeds[base_slice]]
        concept_pool_chunks = [text_embeds_pool[base_slice]]
        text_embed_concepts = torch.cat([chunk for chunk in concept_chunks if chunk.shape[0] > 0], dim=0)
        text_embed_concepts_pool = torch.cat(
            [chunk for chunk in concept_pool_chunks if chunk.shape[0] > 0],
            dim=0,
        )

        if text_embed_concepts.shape[0] == 0:
            text_embed_concepts = text_embed_multi
            text_embed_concepts_pool = text_embed_multi_pool

        at, at_prev = _get_alphas(t)
        dps_t_eps = t.new_tensor(config.dps_t_eps)
        at_eps = scheduler.alphas_cumprod[dps_t_eps.cpu()]

        x_cur = x.detach().to(dtype=torch.float32)
        stats = {}
        unet.requires_grad_(False)

        with torch.autocast(device_type="cuda", enabled=False), torch.enable_grad():
            for cstep in range(num_latent_corrector_steps):

                # ── Jacobian for the full-concept x̂^C_0 ──────────────────────────
                x_cur_C = x_cur.detach().requires_grad_(True)

                noise_pred_base = _predict_noise(
                    torch.cat([x_cur_C, x_cur_C]),
                    t,
                    torch.cat([text_embed_uncond, text_embed_multi], dim=0),
                    torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0),
                )
                # save the original x_cur and noise_pred_base for later use
                if cstep == 0:
                    x_orig = x_cur.detach().clone()
                    noise_pred_base_orig = noise_pred_base.detach().clone()
                
                noise_uncond = noise_pred_base[0:1]
                noise_multi = noise_pred_base[1:2]
                noise_multi_cfg = noise_uncond + config.guidance_scale * (noise_multi - noise_uncond)

                if config.x0_hat_score_source == "multi-cfg": 
                    x0_hat_C = (x_cur_C - (1 - at).sqrt() * noise_multi_cfg) / at.sqrt()
                elif config.x0_hat_score_source == "multi": 
                    x0_hat_C = (x_cur_C - (1 - at).sqrt() * noise_multi) / at.sqrt()
                else:
                    x0_hat_C = (x_cur_C - (1 - at).sqrt() * noise_uncond) / at.sqrt()
                
                z_noise = torch.randn_like(x0_hat_C).detach()
                x_eps_latent_C = at_eps.sqrt() * x0_hat_C + (1 - at_eps).sqrt() * z_noise
                    
                with torch.no_grad():  
                    score_C = _noise2score(
                        _predict_noise(x_eps_latent_C.detach(), dps_t_eps, text_embed_multi, text_embed_multi_pool),
                        at_eps,
                    ) #! very large value at 

                grad_C = torch.autograd.grad(
                    outputs=x0_hat_C,
                    inputs=x_cur_C,
                    grad_outputs=score_C,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                # ── Per-concept Jacobians  Σ_i J^T(x̂^{c_i}_0) · s_{c_i} ──────────
                K = text_embed_concepts.shape[0]
                print(K)
                grad_concepts_sum = torch.zeros_like(grad_C)

                for cc in range(K):
                    
                    # 1. x_t --> x_0|t 
                    
                    x_cur_ci = x_cur.detach().requires_grad_(True)

                    noise_pred_ci = _predict_noise(
                        torch.cat([x_cur_ci, x_cur_ci]),
                        t,
                        torch.cat([text_embed_uncond, text_embed_concepts[cc : cc + 1]], dim=0),
                        torch.cat([text_embed_uncond_pool, text_embed_concepts_pool[cc : cc + 1]], dim=0),
                    )
                    noise_uncond_ci, noise_ci = noise_pred_ci[0:1], noise_pred_ci[1:2]
                    noise_ci_cfg = noise_uncond_ci + config.guidance_scale * (noise_ci - noise_uncond_ci)

                    if config.x0_hat_score_source == "multi-cfg":
                        x0_hat_ci = (x_cur_ci - (1 - at).sqrt() * noise_ci_cfg) / at.sqrt()
                    else:
                        x0_hat_ci = (x_cur_ci - (1 - at).sqrt() * noise_ci) / at.sqrt()
                    
                    # 2. x_0|t --> score(x̂^{c_i}_0|\eps, t=\eps, c_i) 

                    x_eps_latent_ci = at_eps.sqrt() * x0_hat_ci + (1 - at_eps).sqrt() * z_noise
                    with torch.no_grad():
                        score_ci = _noise2score(
                            _predict_noise(
                                x_eps_latent_ci.detach(), dps_t_eps,
                                text_embed_concepts[cc : cc + 1],
                                text_embed_concepts_pool[cc : cc + 1],
                            ),
                            at_eps,
                        )

                    grad_ci = torch.autograd.grad(
                        outputs=x0_hat_ci,
                        inputs=x_cur_ci,
                        grad_outputs=score_ci,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                    grad_concepts_sum = grad_concepts_sum + grad_ci

                # grad = (β)·J^T_C·s_C − β·Σ J^T_{c_i}·s_{c_i}
                grad = (beta) * grad_C - (beta / K) * grad_concepts_sum
                if config.x0_hat_score_source == "multi-cfg":
                    grad = grad / config.guidance_scale 

                if correction_mode == "reverse-sde":

                    cgmin, cgmax = grad.min().item(), grad.max().item()
                    # grad_noise = _score2noise(grad, at)
                    # print("Grad noise: min={:.4f} max={:.4f} norm={:.4f}".format(grad_noise.min().item(), grad_noise.max().item(), grad_noise.norm().item()))
                    # grad = grad.clamp(-5, 5) #! This may need tuning, values are large
                    # grad = grad / (torch.abs(grad).max() + 1e-3) * 5.0  #! try

                    ## 1. Get intermediate x'_{t-1} = x_interm
                    ## Only one denoising step to next_t
                    # if cstep == 0:
                    #     if config.x0_interm_noise_type == "multi-cfg":
                    #         x_interm = scheduler.step(noise_multi_cfg.detach(), t, x_cur.detach(), return_dict=False)[0]
                    #     elif config.x0_interm_noise_type == "multi":
                    #         x_interm = scheduler.step(noise_multi.detach(), t, x_cur.detach(), return_dict=False)[0]
                    #     else:
                    #         x_interm = scheduler.step(noise_uncond.detach(), t, x_cur.detach(), return_dict=False)[0]
                    # else:
                    x_interm = x_cur.detach() # stay at the same timestep
                    
                    # debug
                    correction_tensor = (1 - config.kappa) * noise_multi_cfg.detach() + config.kappa * _score2noise(grad, at).detach()
                    correction_tensor = correction_tensor / (correction_tensor.norm() + 1e-3) * noise_multi.detach().norm()
                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    x_tilde_0 = (1 - eta/num_latent_corrector_steps) * x0_hat_interm + (eta/num_latent_corrector_steps) * grad.detach()

                    ## 2. Update x_cur
                    # x_cur = scheduler.step(correction_tensor, t, x_cur.detach(), return_dict=False)[0].detach()
                    grad = grad.detach() / (grad.norm() + 1e-4) * x0_hat_interm.norm() #! force normalize to avoid magnitude issues
                    x_cur = (x_interm + (float(eta)/num_latent_corrector_steps)  * grad).detach()
                
                    # final outer update at the end of the loop
                    if cstep == num_latent_corrector_steps - 1:
                        net_grad = (x_cur - x_orig).detach()
                        noise_uncond_orig = noise_pred_base_orig[0:1]
                        noise_multi_orig = noise_pred_base_orig[1:2]
                        noise_cfg_orig = noise_uncond_orig + config.guidance_scale * (noise_multi_orig - noise_uncond_orig)
                        if config.x0_interm_noise_type == "multi-cfg":
                            x_interm = scheduler.step(noise_cfg_orig.detach(), t, x_orig.detach(), return_dict=False)[0]
                        elif config.x0_interm_noise_type == "multi":
                            x_interm = scheduler.step(noise_multi_orig.detach(), t, x_orig.detach(), return_dict=False)[0]
                        else:
                            x_interm = scheduler.step(noise_uncond_orig.detach(), t, x_orig.detach(), return_dict=False)[0]
                        x_cur = x_interm + float(eta) * net_grad
                    
                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:
                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")
                        
                        for _name, _tensor in [
                            ("x0_hat_C", x0_hat_C),
                            ("grad_C", grad_C),
                            ("grad", grad),
                            ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_cur", x_cur),
                            ("x_tilde_0", x_tilde_0),
                        ]:
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}.png")
                            # if "grad" in _name:
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_raw.png")
                            viz_latent_raw(_tensor, out_path, title=f"{_name} t={t_val} k={cstep}")
                            # else:
                            out_path = os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}_decoded.png")
                            _tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(_tensor, out_path)
                    
                
                elif correction_mode == "correct-tweedie":
                    #! [Warning] final update step is no implemented yet. 
                    # debug
                    cgmin, cgmax = grad.min().item(), grad.max().item()
                    grad = grad.clamp(-5, 5) #! This may need tuning, values are large
                    # x0_hat_interm = ( x_cur - (1 - at).sqrt() * noise_multi_cfg.detach()) / at.sqrt()
                    noise_interm = _get_interm_noise(cstep, num_latent_corrector_steps, noise_uncond, noise_multi, noise_multi_cfg)
                    x0_hat_interm = _compute_x0_from_xt(x_cur.detach(), at, noise_interm)
                    ## Force normalize?
                    # grad = grad / (grad.norm() + 1e-4) * x0_hat_interm.norm()
                    snr = (1 - at).sqrt() / (at.sqrt()) #Todo: use snr to modulate eta so that it decreases over time
                    x_tilde_0 = (1 - eta) * x0_hat_interm + eta * grad.detach()
                    
                    # x_cur_denoised = ( x_cur - (1 - at).sqrt() * correction_tensor) / at.sqrt()
                    # Force normalize to avoid magnitude issues
                    x_tilde_0 = x_tilde_0 / (x_tilde_0.norm() + 1e-4) * x0_hat_interm.norm()
                    # debug
                    correction_tensor = _x0_to_noise(x_cur, x_tilde_0, at)
                    if config.save_intermediates and viz_state["seed_dir"] is not None and cstep in [0, num_latent_corrector_steps - 1]:
                        t_val = int(t.item())
                        tcorr = viz_state["correction_count"]
                        inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")
                        
                        for _name, _tensor in [
                            ("x0_hat_C", x0_hat_C.detach()),
                            ("grad_C", grad_C),
                            ("grad_concepts_sum", grad_concepts_sum / K),
                            ("grad", grad),
                            ("correction_tensor", correction_tensor),
                            ("x0_hat_interm", x0_hat_interm),
                            ("x_tilde_0", x_tilde_0),
                        ]:
                            if torch.isnan(_tensor).any() or torch.isinf(_tensor).any():
                                print(f"[WARN] {_name} has NaN/Inf")
                            _tensor = 4.0 * _tensor / (_tensor.abs().max() + 1e-8)
                            decode_latent_for_viz(_tensor, os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_k{cstep}_{_name}.png"))
                    
                    # update to new x_cur
                    x_cur = _update_x_cur_from_x_tilde(
                        x_cur, cstep, num_latent_corrector_steps, config.use_cfgpp, config.psi,
                        at, at_prev, x_tilde_0, noise_uncond, noise_multi,
                        forward_noise=noise_multi_cfg, #! noise_multi 
                    )
                else:
                    # x_intermediate = scheduler.step(noise_multi_cfg.detach(), t, x_cur.detach(), return_dict=False)[0]
                    # correction_tensor = grad
                    # x_cur = (x_intermediate + float(eta) * grad).detach()
                    raise ValueError(f"Unknown correction_mode: {correction_mode}")

                stats = {
                    "mode": correction_mode,
                    "correction_tensor": (correction_tensor.min().item(), correction_tensor.max().item()),
                    "grad": (cgmin, cgmax),
                    "eta": float(eta),
                    "x0_hat_interm": (x0_hat_interm.min().item(), x0_hat_interm.max().item()),
                    "x_tilde_0": (x_tilde_0.min().item(), x_tilde_0.max().item()),
                    "x_cur": (x_cur.min().item(), x_cur.max().item()),
                }
                stats_str = " ".join(f"{k}: {_fmt_stats(v)}" for k, v in stats.items())
                print(f"[Correction t={int(t.item())}, k={cstep+1}/{num_latent_corrector_steps}] {stats_str}")
        
        return x_cur.to(dtype=x.dtype).detach(), stats

    def update_x_with_mpgd(x, t, *, correction_mode=None):
        """`correction_mode` overrides config.correction_mode (per-phase hybrid modes)."""
        correction_mode = correction_mode or config.correction_mode
        """MPGD update: no Jacobian — gradient applied directly in x̂_0 space.
        """
        text_embed_uncond = text_embeds[0:1]
        text_embed_multi = text_embeds[1:2]
        text_embed_uncond_pool = text_embeds_pool[0:1]
        text_embed_multi_pool = text_embeds_pool[1:2]

        base_slice = prompt_layout["base_concept_slice"]
        concept_chunks = [text_embeds[base_slice]]
        concept_pool_chunks = [text_embeds_pool[base_slice]]
        text_embed_concepts = torch.cat([chunk for chunk in concept_chunks if chunk.shape[0] > 0], dim=0)
        text_embed_concepts_pool = torch.cat(
            [chunk for chunk in concept_pool_chunks if chunk.shape[0] > 0],
            dim=0,
        )

        if text_embed_concepts.shape[0] == 0:
            text_embed_concepts = text_embed_multi
            text_embed_concepts_pool = text_embed_multi_pool

        at = scheduler.alphas_cumprod[t.cpu()]
        dps_t_eps = t.new_tensor(config.dps_t_eps)
        at_eps = scheduler.alphas_cumprod[dps_t_eps.cpu()]

        x_cur = x.detach().to(dtype=torch.float32)
        stats = {}
        unet.requires_grad_(False)

        with torch.no_grad():
            for _ in range(config.num_latent_corrector_steps):

                # ── Step 1: compute x̂'_0 from x_t ──────────────────────────────
                noise_pred_base = _predict_noise(
                    torch.cat([x_cur, x_cur]),
                    t,
                    torch.cat([text_embed_uncond, text_embed_multi], dim=0),
                    torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0),
                )
                noise_uncond = noise_pred_base[0:1]
                noise_multi = noise_pred_base[1:2]
                noise_multi_cfg = noise_uncond + config.guidance_scale * (noise_multi - noise_uncond)

                if config.x0_hat_score_source == "multi-cfg":
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_multi_cfg) / at.sqrt()
                elif config.x0_hat_score_source == "multi":
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_multi) / at.sqrt()
                else:
                    x0_hat = (x_cur - (1 - at).sqrt() * noise_uncond) / at.sqrt()

                # ── Step 2: compute composed score at x0_hat ────────
                noise_pred_x0 = _predict_noise( torch.cat([x0_hat, x0_hat]), 
                    dps_t_eps,
                    torch.cat([text_embed_uncond, text_embed_multi], dim=0),
                    torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0),
                )
                noise_uncond_x0 = noise_pred_x0[0:1]
                noise_C_x0 = noise_pred_x0[1:2]
                if config.ddim_forward_type in ["cfg", "uncond"]:
                    noise_C_x0 = noise_uncond_x0 + config.guidance_scale * (noise_C_x0 - noise_uncond_x0) 
                
                score_C_x0 = _noise2score(
                    noise_C_x0,
                    at_eps,
                )

                K = text_embed_concepts.shape[0]
                # print(K)
                score_concepts_sum = torch.zeros_like(score_C_x0)
                for cc in range(K):
                    noise_ci_x0 = _predict_noise(
                            x0_hat,
                            dps_t_eps,
                            text_embed_concepts[cc : cc + 1],
                            text_embed_concepts_pool[cc : cc + 1],
                        )
                    if config.ddim_forward_type in ["cfg", "uncond"]:
                        noise_ci_x0 = noise_uncond_x0 + config.guidance_scale * (noise_ci_x0 - noise_uncond_x0)
                    score_ci_x0 = _noise2score(noise_ci_x0, at_eps)
                    
                    score_concepts_sum = score_concepts_sum + score_ci_x0

                
                composed_score = score_C_x0 - (1.0 / K) * score_concepts_sum
                #Todo: projection of composed_score onto multi manifold
                x_tilde_0 = x0_hat + config.beta * composed_score
                # debug
                stats = {"x0_hat": (x0_hat.min().item(), x0_hat.max().item()), 
                         "x_tilde_0": (x_tilde_0.min().item(), x_tilde_0.max().item())
                         }
                
                # Force normalize
                x_tilde_0 = x_tilde_0 / x_tilde_0.norm() * x0_hat.norm()


                # ── Step 3: manual DDIM step using noise predicted at x_cur  ─────────
                t_scalar = int(t.item())
                ts_list = scheduler.timesteps.tolist()
                cur_idx = next((i for i, v in enumerate(ts_list) if int(v) == t_scalar), -1)
                if cur_idx == -1 or cur_idx == len(ts_list) - 1:
                    at_prev = scheduler.final_alpha_cumprod.to(dtype=torch.float32)
                else:
                    at_prev = scheduler.alphas_cumprod[int(ts_list[cur_idx + 1])].to(dtype=torch.float32)
                    
                if config.ddim_forward_type == "cfg":
                    assert config.x0_hat_score_source == "multi-cfg", "ddim_forward_type=cfg requires x0_hat_score_source=multi-cfg"
                    x_cur = (at_prev.sqrt() * x_tilde_0 + (1 - at_prev).sqrt() * noise_multi_cfg).detach()  
                elif config.ddim_forward_type == "uncond":
                    assert (config.guidance_scale < 1.01 and config.x0_hat_score_source == "multi-cfg"), "uncond forward type requires guidance_scale <= 1.0 and x0_hat_score_source=multi-cfg"
                    x_cur = (at_prev.sqrt() * x_tilde_0 + (1 - at_prev).sqrt() * noise_uncond).detach()
                else:
                    assert config.x0_hat_score_source == "multi", "multi ddim_forward_type requires x0_hat_score_source = multi"
                    x_cur = (at_prev.sqrt() * x_tilde_0 + (1 - at_prev).sqrt() * noise_multi).detach()

                if config.save_intermediates and viz_state["seed_dir"] is not None:
                    t_val = int(t.item())
                    tcorr = viz_state["correction_count"]
                    inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")
                    composed_score = score_C_x0 - (1.0 / K) * score_concepts_sum
                    for _name, _tensor in [
                        ("x0_hat", x0_hat),
                        ("score_C_x0", score_C_x0),
                        ("score_concepts_sum", score_concepts_sum),
                        ("composed_score", composed_score),
                        ("x_tilde_0", x_tilde_0),
                        ("x_cur", x_cur),
                    ]:
                        decode_latent_for_viz(_tensor, os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_mpgd_{_name}.png"))

                stats.update({
                    "mode": "mpgd",
                    "composed_grad": (composed_score.min().item(), composed_score.max().item()),
                    "eta": float(config.eta),
                })

        return x_cur.to(dtype=x.dtype).detach(), stats

    def denoise_step(x, t, step, correction_step_set, algo_version="option1", correction_mode=None):
        """`algo_version` / `correction_mode` come from the caller, not from config.

        `correction_mode` is the mode used by non-hybrid algos; the hybrids ignore it and
        use `algo1_correction_mode` / `algo2_correction_mode` for their two phases.
        """
        correction_mode = correction_mode or config.correction_mode
        text_embed_uncond = text_embeds[0].unsqueeze(0)
        text_embed_multi = text_embeds[1].unsqueeze(0)
        text_embed_uncond_pool = text_embeds_pool[0].unsqueeze(0)
        text_embed_multi_pool = text_embeds_pool[1].unsqueeze(0)
        at, at_prev = _get_alphas(t.cpu())

        denoised_latent = None
        if config.perform_latent_correction and step in correction_step_set:
            print("Bef Corr: x stats: min={:.4f} max={:.4f} norm={:.4f}".format(x.min().item(), x.max().item(), x.norm().item()))
            boosted_grad_ts = []
            if config.nts_boosted_grad > 0:
                boosted_grad_ts = scheduler.timesteps[:config.nts_boosted_grad]
            if config.nts_to_init_correct > 0 and t in scheduler.timesteps[:config.nts_to_init_correct]:
                num_latent_corrector_steps = config.init_latent_corrector_steps
            else:
                # boosted_grad_ts = None
                num_latent_corrector_steps = config.num_latent_corrector_steps
            
            if algo_version == "option2":
                # eta_t = max(config.eta / (step + 1), 0.05) 
                eta_t =  config.eta * (1 - config.eta_schedule_factor * step / config.n_timesteps) *  at_prev.sqrt() #! try this snr based adaptive eta. Need to maintain grad.max() <= x_0t.max()
                x, dps_stats = update_x_with_dps_option2(x, t, num_latent_corrector_steps, eta=eta_t, beta=config.beta, correction_mode=correction_mode)

            elif algo_version == "mpgd":
                x, dps_stats = update_x_with_mpgd(x, t, num_latent_corrector_steps)

            elif algo_version == "energy":
                if correction_mode == "correct-tweedie":
                    eta_t = config.eta * (1 - config.eta_schedule_factor * step / config.n_timesteps) * 1.0
                else:
                    eta_t = config.eta * (1 - config.eta_schedule_factor * step / config.n_timesteps) * 1.0 #at_prev.sqrt()
                x, dps_stats = update_x_with_energy_grad(x, t, num_latent_corrector_steps, eta=eta_t, beta=config.beta, boosted_grad_ts=boosted_grad_ts, correction_mode=correction_mode)

            elif algo_version == "energy-opt2":
                if correction_mode == "correct-tweedie":
                    eta_t = config.eta * (1 - config.eta_schedule_factor * step / config.n_timesteps) * 1.0
                else:
                    eta_t = config.eta * (1 - config.eta_schedule_factor * step / config.n_timesteps) * 1.0 #at_prev.sqrt()
                x, dps_stats = update_x_with_energy_grad_opt2(x, t, num_latent_corrector_steps, eta=eta_t, beta=config.beta, boosted_grad_ts=boosted_grad_ts, correction_mode=correction_mode)

            elif algo_version == "energy-netgrad":
                eta_t = config.eta * (1 - config.eta_schedule_factor * step / config.n_timesteps)
                x, dps_stats = update_x_with_energy_eucl_netgrad(
                    x, t, num_latent_corrector_steps, eta=eta_t, beta=config.beta,
                    correction_mode=correction_mode,
                )

            elif algo_version == "energy-hybrid-1-2":
                eta_factor = 1.0 #if config.correction_mode == "correct-tweedie" else at_prev.sqrt() #! the factor is moved inside the algo funcitons. 
                if step < config.num_steps_first_algo:
                    eta_t = config.eta_opt1 * (1 - config.eta_schedule_factor * step / config.n_timesteps) * eta_factor
                    x, dps_stats = update_x_with_energy_grad(x, t, num_latent_corrector_steps, eta=eta_t, beta=config.beta_opt1, boosted_grad_ts=boosted_grad_ts, correction_mode=algo1_correction_mode)
                else:
                    eta_t = config.eta_opt2 * (1 - config.eta_schedule_factor * step / config.n_timesteps) * eta_factor
                    x, dps_stats = update_x_with_energy_grad_opt2(x, t, num_latent_corrector_steps, eta=eta_t, beta=config.beta_opt2, boosted_grad_ts=boosted_grad_ts, correction_mode=algo2_correction_mode)

            elif algo_version == "hybrid-1-2":
                if step < config.num_steps_first_algo:
                    x, dps_stats = update_x_with_dps(
                        x, t, num_latent_corrector_steps,
                        eta=config.eta_opt1, beta=config.beta_opt1,
                        correction_mode=algo1_correction_mode,
                    )
                else:
                    x, dps_stats = update_x_with_dps_option2(
                        x, t, num_latent_corrector_steps,
                        eta=config.eta_opt2, beta=config.beta_opt2,
                        correction_mode=algo2_correction_mode,
                    )
            else:
                eta_t =  config.eta * (1 - config.eta_schedule_factor * step / config.n_timesteps) * at_prev.sqrt()
                x, dps_stats = update_x_with_dps(x, t, num_latent_corrector_steps, eta=eta_t, beta=config.beta, correction_mode=correction_mode)
            print("Aft Corr: x stats: min={:.2f} max={:.2f} norm={:.2f}".format(x.min().item(), x.max().item(), x.norm().item()))

            t_val = int(t.item())
            for k, v in dps_stats.items():
                if isinstance(v, (int, float, tuple)):
                    stats_history.setdefault(k, []).append((t_val, v))

            stats_str = " ".join(f"{k}: {_fmt_stats(v)}" for k, v in dps_stats.items())
            print(f"[Correction t={int(t.item())}] {stats_str}")
            denoised_latent = x


        if step not in correction_step_set:
            with torch.no_grad():
                latent_model_input = torch.cat([x, x])
                text_embed = torch.cat([text_embed_uncond, text_embed_multi], dim=0)
                text_embed_pool = torch.cat([text_embed_uncond_pool, text_embed_multi_pool], dim=0)
                noise_pred = _predict_noise(latent_model_input, t, text_embed, text_embed_pool)
                noise_pred_uncond = noise_pred[:1]
                noise_pred_cond = noise_pred[1:2]
                noise_pred_cfg = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            denoised_tweedie = (x - (1 - at).sqrt() * noise_pred_cfg) / at.sqrt()
            if config.use_cfgpp:
                denoised_latent = at_prev.sqrt() * denoised_tweedie + (1 - at_prev).sqrt() * noise_pred_uncond
            else:
                denoised_latent = scheduler.step(noise_pred_cfg, t, x, return_dict=False)[0] 
            print("No Correction | time:{} x_cur stats: min={:.2f} max={:.2f} norm={:.2f}".format(int(t.item()), x.min().item(), x.max().item(), x.norm().item()))
            print("No Correction \t\t | denoised_latent stats: min={:.2f} max={:.2f} norm={:.2f}".format(denoised_latent.min().item(), denoised_latent.max().item(), denoised_latent.norm().item()))

            t_val = int(t.item())
            no_corr_stats = {
                "denoised_tweedie": (denoised_tweedie.min().item(), denoised_tweedie.max().item()),
                "denoised_latent": (denoised_latent.min().item(), denoised_latent.max().item()),
            }
            # viz
            if config.save_intermediates and viz_state["seed_dir"] is not None:
                tcorr=len(correction_step_set)
                inter_dir = os.path.join(viz_state["seed_dir"], "intermediates", f"tcorr{tcorr}")
                os.makedirs(inter_dir, exist_ok=True)
                tcorr = viz_state["correction_count"]
                decode_tensor = 4.0 * denoised_tweedie / (denoised_tweedie.abs().max() + 1e-8)
                
                decode_latent_for_viz(
                    decode_tensor,
                    os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_nocorr_denoised_tweedie_decoded.png"),
                )
                viz_latent_raw(
                    denoised_tweedie,
                    os.path.join(inter_dir, f"tcorr{tcorr}_t{t_val}_nocorr_denoised_tweedie_raw.png"),                    
                )

            for k, v in no_corr_stats.items():
                stats_history.setdefault(k, []).append((t_val, v))
            
            print(" ")
        if t == scheduler.timesteps[-1]:
            return denoised_tweedie
        return denoised_latent

    def decode_latent_for_viz(latent, filename):
        with torch.no_grad():
            x = latent.detach().clone().float()
            has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
            has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(x.device, x.dtype)
                latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(x.device, x.dtype)
                x = x * latents_std / vae.config.scaling_factor + latents_mean
            else:
                x = x / vae.config.scaling_factor
            decoded = vae.decode(x.half(), return_dict=False)[0]
        image = image_processor.postprocess(decoded, output_type="pil")
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        image[0].save(filename)

    def viz_latent_raw(tensor, filename, title=""):
        x = tensor.detach().float().cpu()
        if x.dim() == 4:
            x = x[0]
        x3 = x[:3]
        vmin, vmax = x3.min().item(), x3.max().item()
        img = (x3 - vmin) / (vmax - vmin + 1e-8)
        img = img.permute(1, 2, 0).numpy()
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{title} min={vmin:.3f} max={vmax:.3f}")
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    import builtins
    builtins.decode_latent_for_viz = decode_latent_for_viz
    builtins.viz_latent_raw = viz_latent_raw

    def sample_loop(x, correction_step_set):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling")):
                x = denoise_step(
                    x, t, i, correction_step_set,
                    algo_version=config.algo_version,
                    correction_mode=config.correction_mode,
                )

            with torch.no_grad():
                has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
                has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(x.device, x.dtype)
                    latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(x.device, x.dtype)
                    x = x * latents_std / vae.config.scaling_factor + latents_mean
                else:
                    x = x / vae.config.scaling_factor
                decoded_latent = vae.decode(x, return_dict=False)[0]
            image = image_processor.postprocess(decoded_latent, output_type="pil")
        return image

    def plot_diagnostics(stats_history, save_dir, tag):
        series = {}
        for key, points in stats_history.items():
            if not points:
                continue
            pts = sorted(points, key=lambda p: p[0], reverse=True)  # T -> 0
            ts = [p[0] for p in pts]
            vals = [p[1][1] if isinstance(p[1], tuple) else p[1] for p in pts]
            series[key] = (ts, vals)

        if not series:
            return

        diag_dir = os.path.join(save_dir, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        fig, axes = plt.subplots(len(series), 1, figsize=(8, 3 * len(series)), sharex=True)
        if len(series) == 1:
            axes = [axes]
        for ax, (key, (ts, vals)) in zip(axes, series.items()):
            ax.plot(ts, vals, marker="o")
            ax.set_ylabel(key)
            ax.grid(alpha=0.3)
        axes[-1].invert_xaxis()  # axes share x (sharex=True); invert once after all data is plotted
        axes[-1].set_xlabel("Sampling timestep t (T -> 0)")
        fig.suptitle(f"Correction diagnostics: {tag}")
        fig.tight_layout()
        path = os.path.join(diag_dir, f"{tag}_diagnostics.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Diagnostic plot -> {path}]")

# ------------------------------------- Code starts here -------------------------------------

    if config.run_mode == "debug":
        print(f"Prompt: {config.prompt_orig}")
        print(f"Inferred concepts: {prompt_layout['concept_items']}")
        print(f"Composer prompts: {all_prompts}")
        print(f"Base concept slice: {prompt_layout['base_concept_slice']}")
        if config.use_attribute_guidance:
            print(f"Attribute slice: {prompt_layout['attribute_slice']}")
        print(f"x0_hat_score_source: {config.x0_hat_score_source}")
        print(f"Seeds: {seed_values}")
        print(f"Correction count runs: {correction_counts}")
        if config.use_cfgpp:
            assert config.guidance_scale <= 1.0, "cfgpp needs guidance_scale <= 1.0 to avoid divergence"

        for seed in seed_values:
            for correction_count in correction_counts:
                # Re-seed per run: each sample_loop consumes the global RNG stream
                # (energy tau/eps draws, corrector noise), so seeding once per seed
                # would leave every run after the first starting mid-stream.
                seed_everything(seed)
                base_normal = torch.randn(
                    1,
                    config.latent_channels,
                    config.resolution_h // 8,
                    config.resolution_w // 8,
                    device=unet.device,
                ) * scheduler.init_noise_sigma

                correction_steps = list(range(correction_count))
                correction_step_set = set(correction_steps)
                correction_timesteps = [int(scheduler.timesteps[idx].item()) for idx in correction_steps]
                print(f"Seed: {seed}")
                print(f"Correction steps (1-based): {[idx + 1 for idx in correction_steps]}")
                print(f"Correction timesteps: {correction_timesteps}")

                seed_dir = os.path.join(
                    config.output_path_all,
                    config.run_mode,
                    config.algo_version,
                    "use_cfgpp{}-{}".format(config.use_cfgpp, config.guidance_scale),
                    corr_mode_str(),
                    bounded_name(
                        f"nsteps{config.n_timesteps}_"
                        f"initnCorrSteps{config.init_latent_corrector_steps}_"
                        f"ntsBoostGrad{config.nts_boosted_grad}_"
                        f"nCorrSteps{config.num_latent_corrector_steps}_"
                        f"base{config.x0_hat_score_source}_"
                        f"x0interm{config.x0_interm_noise_type}_"
                        f"ddimFwd{config.ddim_forward_type}_"
                        f"teps{config.dps_t_eps}_"
                        f"kappa{config.kappa}_{eta_beta_str()}"
                        f"psi{config.psi}_"
                        + energy_str()
                        + extra_knobs_str(include_psi=False)
                    ),
                    prompt_dir_name,
                    f"run-{config.run_id}",
                    f"seed-{seed}",
                )
                os.makedirs(seed_dir, exist_ok=True)
                # dump config
                config_path = os.path.join(seed_dir, "config.yaml")
                with open(config_path, "w") as f:
                    yaml.safe_dump(asdict(config), f, sort_keys=False)
                print(f"Saved config: {config_path}")

                
                viz_state["seed_dir"] = seed_dir
                viz_state["correction_count"] = correction_count
                viz_state["seed"] = seed
                stats_history.clear()
                image = sample_loop(base_normal.clone(), correction_step_set)
                plot_diagnostics(stats_history, seed_dir, f"tcorr{correction_count}")
                save_name = f"tcorr{correction_count}.png"
                save_path = os.path.join(seed_dir, save_name)
                image[0].save(save_path)
                print(f"Saved image: {save_path}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    main()
