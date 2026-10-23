"""FLUX.2 VAE with Studio's 5D, normalized 32-channel latent contract."""
import torch
from modeling.moss_image.vae import load_flux2_vae, patchify_latents, unpatchify_latents
from ...vae import VAEWrapper


class Flux2Codec(torch.nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def encode(self, pixels, scale=None):
        if pixels.ndim != 5 or pixels.shape[2] != 1:
            raise ValueError('MOSS VAE supports images only (B,3,1,H,W)')
        with torch.autocast(pixels.device.type, enabled=False):
            latent = self.autoencoder.encode_for_dit(pixels[:, :, 0].float())
            return unpatchify_latents(latent).unsqueeze(2)

    def decode(self, latent, scale=None):
        if latent.ndim != 5 or latent.shape[2] != 1:
            raise ValueError('MOSS VAE expects (B,32,1,H,W)')
        with torch.autocast(latent.device.type, enabled=False):
            packed = patchify_latents(latent[:, :, 0].float())
            return self.autoencoder.decode_from_dit(packed).unsqueeze(2)


def load_vae(path, device, *, tiling='auto'):
    model = Flux2Codec(load_flux2_vae(path, torch.float32, device)).eval().requires_grad_(False)
    # Normalization is owned by Flux2Codec (128-channel BN after 2x2 packing).
    return VAEWrapper(model, torch.zeros(32, device=device), torch.ones(32, device=device), tiling)
