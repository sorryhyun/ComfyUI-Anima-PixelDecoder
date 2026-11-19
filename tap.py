"""The feature tap: what the pixel head reads, kept from the last sampling step.

The head reads the input of the DiT's ``final_layer.linear`` (``hidden_size`` per token, after
the adaLN-modulated LayerNorm) and x_t at the last step, CFG-combined as f_u + g·(f_c − f_u).
Three hooks on a cloned MODEL, all bound to the live modules at call time (a Block Compile
rebuild downstream is fine):

- ``CALC_COND_BATCH`` wrapper: opens a step record (σ) on every sampler step, cached or not.
- ``DIFFUSION_MODEL`` wrapper: for one actual forward, a pre-hook on the live
  ``final_layer.linear`` keeps its input; x and ``cond_or_uncond`` come from the call.
  Spectrum's cached steps run ``final_layer`` without the DiT forward, so a step that ends with
  no forward of its own marks the tail as cached.
- post-CFG function: the CFG scale of the step, and whether a ``sampler_cfg_function``
  (CFG++, rescale, …) replaced the plain combine.

Only the last step with an actual forward is kept.
"""

from __future__ import annotations

import logging

import comfy.patcher_extension
import torch

logger = logging.getLogger(__name__)

KEY = "anima_pixel_tap"


class TapState:
    def __init__(self):
        self.step: dict | None = None  # the step in progress / the last step
        self.last: dict | None = None  # the last step with an actual forward

    def open_step(self, sigma: float) -> None:
        self.step = {"sigma": sigma, "forwards": [], "cond_scale": None, "cfg_fn": False}

    def close_step(self) -> None:
        if self.step is not None and self.step["forwards"]:
            self.last = self.step

    @property
    def tail_cached(self) -> bool:
        return self.step is not None and not self.step["forwards"]

    def cond_scale(self) -> float | None:
        """The last step's CFG scale (a cached tail step still runs the CFG combine)."""
        for s in (self.step, self.last):
            if s is not None and s["cond_scale"] is not None:
                return s["cond_scale"]
        return None

    def features(
        self, cond_scale: float | None
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(f, x_t, σ) of the last actual step: f (B, T, gh, gw, D) fp32, CFG-combined;
        x_t (B, C, T, h, w) as the DiT saw it."""
        if self.last is None:
            raise RuntimeError(
                "Anima Pixel Decode: no forward was recorded. Put Anima Pixel Head Tap on "
                "the MODEL that goes into the sampler."
            )
        parts: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}
        for fw in self.last["forwards"]:
            cou = fw["cond_or_uncond"]
            for kind, f, x in zip(
                cou, fw["f"].chunk(len(cou)), fw["x"].chunk(len(cou))
            ):
                parts.setdefault(kind, []).append((f, x))
        if len(parts[0]) != 1 or len(parts[1]) > 1:
            raise RuntimeError(
                "Anima Pixel Decode: the last step ran "
                f"{len(parts[0])} positive and {len(parts[1])} negative conditionings; "
                "the head reads one of each (combined or area conditionings are not "
                "supported)."
            )
        f_c, x = parts[0][0]
        f_c = f_c.float()
        if parts[1]:
            if cond_scale is None:
                raise RuntimeError(
                    "Anima Pixel Decode: the CFG scale of the last step was not recorded"
                )
            f_u = parts[1][0][0].float()
            f = f_u + cond_scale * (f_c - f_u)
        else:
            f = f_c
        return f, x, self.last["sigma"]


def _calc_cond_batch_wrapper(state: TapState):
    def wrapper(executor, model, conds, x_in, timestep, model_options):
        state.open_step(float(timestep.flatten()[0]))
        try:
            return executor(model, conds, x_in, timestep, model_options)
        finally:
            state.close_step()

    return wrapper


def _diffusion_model_wrapper(state: TapState):
    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        step = state.step
        to = kwargs.get("transformer_options", {})
        cou = to.get("cond_or_uncond")
        if step is None or not cou:  # a call outside the sampler's cond batching
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        record = {"x": x.detach(), "f": None, "cond_or_uncond": list(cou)}

        def keep(_module, inputs):
            if record["f"] is None:
                record["f"] = inputs[0].detach()

        handle = executor.class_obj.final_layer.linear.register_forward_pre_hook(keep)
        try:
            out = executor(x, timesteps, context, fps, padding_mask, **kwargs)
        finally:
            handle.remove()
        if record["f"] is not None:
            step["forwards"].append(record)
        return out

    return wrapper


def _post_cfg(state: TapState):
    def post_cfg(args):
        if state.step is not None:
            state.step["cond_scale"] = float(args["cond_scale"])
            state.step["cfg_fn"] = "sampler_cfg_function" in args.get("model_options", {})
        return args["denoised"]

    post_cfg._anima_pixel_tap = True
    return post_cfg


def apply_tap(model):
    """A clone of ``model`` with the tap; re-tapping a tapped model replaces the old tap."""
    m = model.clone()
    state = TapState()
    for wtype in (
        comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
    ):
        m.remove_wrappers_with_key(wtype, KEY)
    fns = m.model_options.get("sampler_post_cfg_function", [])
    m.model_options["sampler_post_cfg_function"] = [
        fn for fn in fns if not getattr(fn, "_anima_pixel_tap", False)
    ]
    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
        KEY,
        _calc_cond_batch_wrapper(state),
    )
    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        KEY,
        _diffusion_model_wrapper(state),
    )
    m.set_model_sampler_post_cfg_function(_post_cfg(state))
    m.set_attachments(KEY, state)
    return m


def get_state(model) -> TapState | None:
    return model.get_attachment(KEY)
