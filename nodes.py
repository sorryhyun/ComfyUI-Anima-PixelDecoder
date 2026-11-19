"""Anima Pixel Head Tap (MODEL → MODEL) and Anima Pixel Decode (→ IMAGE)."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

import comfy.model_management
import folder_paths
import torch
import torch.nn.functional as F

from .tap import apply_tap, get_state

logger = logging.getLogger(__name__)

FOLDER = "pixel_heads"
if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        FOLDER, os.path.join(folder_paths.models_dir, FOLDER)
    )
folder_paths.folder_names_and_paths[FOLDER][1].add(".safetensors")


def _vendor():
    """``_vendor/pixel_head.py``: a copy of anima_lora's ``library/anima/pixel_head.py``
    (``make vendor-sync`` there), imported under a private name."""
    name = "comfyui_anima_pixel_decoder_pixel_head"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(__file__), "_vendor", "pixel_head.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ph = _vendor()


def _head_names() -> list[str]:
    """Files in ``models/pixel_heads/``, and files in the VAE folders named ``*pixel_head*``."""
    names = list(folder_paths.get_filename_list(FOLDER))
    names += [
        n
        for n in folder_paths.get_filename_list("vae")
        if "pixel_head" in os.path.basename(n) and n not in names
    ]
    return names


def _head_path(name: str) -> str:
    path = folder_paths.get_full_path(FOLDER, name) or folder_paths.get_full_path(
        "vae", name
    )
    if path is None:
        raise FileNotFoundError(f"pixel head {name!r} not found")
    return path


_HEADS: dict[str, tuple[float, torch.nn.Module]] = {}


def _load_head(path: str, device) -> torch.nn.Module:
    mtime = os.path.getmtime(path)
    hit = _HEADS.get(path)
    if hit is None or hit[0] != mtime:
        hit = (mtime, ph.load_pixel_head(path))
        _HEADS.clear()  # one head resident
        _HEADS[path] = hit
    return hit[1].to(device)


@torch.no_grad()
def pixel_decode(head, state, latent: torch.Tensor) -> torch.Tensor:
    """IMAGE (B, H, W, 3) in [0, 1] from the tap's last step, on the intermediate device."""
    f, x, sigma = state.features(state.cond_scale())
    step = state.last
    if ph.out_of_range(sigma, head.sigma_ranges):
        ranges = ", ".join(
            f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}" for lo, hi in head.sigma_ranges
        )
        logger.warning(
            f"Anima Pixel Decode: last-step σ {sigma:.4f} is outside the σ the head was "
            f"trained at ({ranges}); it still decodes, quality there is unmeasured"
        )
    if state.tail_cached:
        logger.warning(
            "Anima Pixel Decode: the last step was a cached (Spectrum) step; decoding the "
            f"last actual forward (σ {sigma:.4f})"
        )
    if step["cfg_fn"] or (state.step is not None and state.step["cfg_fn"]):
        logger.warning(
            "Anima Pixel Decode: a sampler_cfg_function replaced the CFG combine; the head "
            "combines the features as f_u + g·(f_c − f_u)"
        )

    if x.ndim == 5:
        x = x[:, :, 0]
    if f.ndim == 5:
        f = f[:, 0]
    lat = latent if latent.ndim == 4 else latent[:, :, 0]
    if x.shape[0] != lat.shape[0] or x.shape[-2:] != lat.shape[-2:]:
        raise RuntimeError(
            f"Anima Pixel Decode: the tapped step's latent {tuple(x.shape)} does not match "
            f"the LATENT input {tuple(lat.shape)}; decode the output of the sampler that "
            "ran the tapped MODEL"
        )
    h, w = x.shape[-2:]
    dev = next(head.parameters()).device
    x = x.to(dev).float()
    f = f.to(dev)
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
    out = []
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for b in range(f.shape[0]):
            out.append(head(f[b], ph.lat_patches(x[b])).float())
    px = torch.stack(out)[:, :, : h * 8, : w * 8].clamp(-1, 1)
    img = ((px + 1) / 2).permute(0, 2, 3, 1)
    return img.to(comfy.model_management.intermediate_device(), torch.float32)


class AnimaPixelHeadTap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima/pixel decoder"
    DESCRIPTION = (
        "Keeps what the Anima pixel head reads (the input of the DiT's final_layer.linear "
        "and x_t, per CFG branch) from the sampler's last step. Put it on the MODEL that "
        "goes into the KSampler (stock or Spectrum), then decode the sampler's LATENT with "
        "Anima Pixel Decode instead of VAE Decode. Changes no output."
    )

    def patch(self, model):
        return (apply_tap(model),)


class AnimaPixelDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The tapped MODEL the sampler ran."}),
                "samples": ("LATENT", {"tooltip": "The sampler's output."}),
                "head_name": (_head_names(),),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "anima/pixel decoder"
    DESCRIPTION = (
        "Decodes the last sampling step to RGB with the Anima pixel head instead of the "
        "VAE. Trained at Euler shift 3 with 20–50 steps on the base DiT (CFG 4) and at the "
        "4-step turbo (CFG 1); elsewhere it decodes with a warning in the log."
    )

    def decode(self, model, samples, head_name):
        state = get_state(model)
        if state is None:
            raise RuntimeError(
                "Anima Pixel Decode: the MODEL has no tap; put Anima Pixel Head Tap before "
                "the sampler and connect the same MODEL here"
            )
        head = _load_head(_head_path(head_name), comfy.model_management.get_torch_device())
        return (pixel_decode(head, state, samples["samples"]),)


NODE_CLASS_MAPPINGS = {
    "AnimaPixelHeadTap": AnimaPixelHeadTap,
    "AnimaPixelDecode": AnimaPixelDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaPixelHeadTap": "Anima Pixel Head Tap",
    "AnimaPixelDecode": "Anima Pixel Decode",
}
