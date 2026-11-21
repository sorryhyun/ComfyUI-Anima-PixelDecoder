# ComfyUI Anima Pixel Decoder

Decodes [Anima](https://huggingface.co/circlestone-labs/Anima) renders to RGB straight from the
DiT's last-layer features with a 25M-parameter pixel head, in place of the VAE decode. The
head reads the input of the DiT's `final_layer.linear` at the last sampling step (CFG-combined
as f_u + g·(f_c − f_u)) and x_t there, and writes a 16×16 RGB patch per token.

Weights: [sorryhyun/anima-pixel-decoder](https://huggingface.co/sorryhyun/anima-pixel-decoder)
(`anima_pixel_head.safetensors`). Put the file in `models/pixel_heads/`, or anywhere in a VAE
folder under a name containing `pixel_head`.

## Nodes

| Node | In → out | |
|---|---|---|
| **Anima Pixel Head Tap** | MODEL → MODEL | Keeps the last step's features and x_t per CFG branch. Changes no output. |
| **Anima Pixel Decode** | MODEL, LATENT, head → IMAGE | Use in place of VAE Decode. `model` is the tapped MODEL the sampler ran, `samples` is the sampler's output. |

```
Load Diffusion Model → (LoRA / Adapter Loader) → (Block Compile) → Anima Pixel Head Tap ─┬→ KSampler → Anima Pixel Decode → Save Image
                                                                                     └──────────────────↗ (model)
```

The tap sits on the DiT, not on a sampler: the stock KSampler and **KSampler (Spectrum)** both
work. Spectrum's last steps are actual forwards; if the last step was a cached one, the node
decodes the last actual forward and says so in the log.

## ×2 output

`anima_pixel_head_x2.safetensors` (same weights repo) decodes and upscales 2× in one readout:
a 1024² render comes out at 2048². Pick it in `head_name`: the file decides the scale, so the ×2
file returns 2× and `anima_pixel_head.safetensors` returns ×1 (there is no scale input).
It was trained on the base DiT only (below); on the 4-step turbo it decodes with a warning.

## Where the head was trained

- Base DiT (`anima-base-v1.0`): Euler, shift 3, 20 … 50 steps (last-step σ 0.058 … 0.136),
  CFG 4, Spectrum included. Both files.
- Turbo (the ×1 file only): the 4-step turbo LoRA of
  [anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step), Euler 4 steps,
  shift 3 (`simple` scheduler), CFG 1 (last-step σ 0.5). The head was trained with that repo's
  `anima_turbo_4step_v4_longer.safetensors` (the previous head, `old/anima_pixel_head.safetensors`
  in the weights repo, with its `old/anima_turbo_4step.safetensors`).
- fp32 residual stream: the Anima Adapter Loader's `stream = fp32` (or `AnimaStreamFP32`) from
  [ComfyUI-Anima_lora-Adapter](https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter).

Elsewhere the head still decodes and the log warns (σ outside the trained points, a cached last
step, a `sampler_cfg_function` such as CFG++ replacing the plain CFG combine). Combined or area
conditionings (more than one positive or negative at the last step) are refused.

SMC-CFG from [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)
is the one `sampler_cfg_function` the tap follows: it keeps the velocity SMC-CFG adds to the plain
combine at the last step and shifts the head's features by the minimum-norm amount that makes
`final_layer.linear` return that velocity. The head was not trained on SMC-CFG renders.

## Code

`_vendor/pixel_head.py` is a copy of `library/anima/pixel_head.py` from
[anima_lora](https://github.com/sorryhyun/anima_lora), written by `make vendor-sync` there; edit
it there. `tap.py` holds the ComfyUI hooks: a `CALC_COND_BATCH` wrapper (one record per sampler
step), a `DIFFUSION_MODEL` wrapper with a pre-hook on the live `final_layer.linear` for the call,
and a post-CFG function for the CFG scale (and, under SMC-CFG, the velocity residual).

## License

Code: MIT. The head weights are a Derivative of Anima under the CircleStone Labs Non-Commercial
License (see the weights repo).
