"""One deferred VAE decoder for teacher and PC latent samples."""

from pathlib import Path

import torch


class AFHQDecoder:
    def __init__(self, vae=None, *, path="checkpoints/metric_flow/vae", device="cpu"):
        self.path, self.device = Path(path), torch.device(device)
        self.vae = vae

    def _load(self):
        if self.vae is None:
            if not (self.path / "config.json").is_file():
                raise FileNotFoundError(
                    f"Place the CompVis/stable-diffusion-v1-4 VAE files under {self.path}"
                )
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(
                str(self.path), local_files_only=True, use_safetensors=True
            )
        self.vae.to(self.device).eval().requires_grad_(False)
        return self.vae

    @torch.no_grad()
    def raw(self, latents):
        if tuple(latents.shape[1:]) not in ((1024,), (4, 16, 16)):
            raise ValueError(
                "AFHQ decoder expects [B,1024] or [B,4,16,16] latent means"
            )
        vae = self._load()
        dtype = next(vae.parameters()).dtype
        # Upstream caches posterior means WITHOUT the Stable Diffusion scale.
        return vae.decode(
            latents.reshape(-1, 4, 16, 16).to(self.device, dtype=dtype)
        ).sample

    def __call__(self, latents):
        # VaeImageProcessor.postprocess denormalization, returned as tensors.
        return (self.raw(latents) / 2 + 0.5).clamp(0, 1)
