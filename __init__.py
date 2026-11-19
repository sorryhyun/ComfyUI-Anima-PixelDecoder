"""Anima pixel decoder ComfyUI nodes.

* ``AnimaPixelHeadTap`` — MODEL -> MODEL. Keeps the last sampling step's DiT features.
* ``AnimaPixelDecode`` — (MODEL, LATENT, head) -> IMAGE, in place of VAE Decode.

The head code is ``_vendor/pixel_head.py``, a copy of anima_lora's
``library/anima/pixel_head.py`` written by ``make vendor-sync`` there.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
