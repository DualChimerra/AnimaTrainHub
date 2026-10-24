"""MOSS Euler sampling with simple / SGM uniform schedules and standard CFG.

Simple schedule and CFG order adapted from ComfyUI via WalkingMeatAxolotl/AnimaLoraStudio #616; see THIRD_PARTY_NOTICES.md.
"""
import torch
from modeling.moss_image.vae import unpatchify_latents

def _sigma_grid(count: int, shift: float, total: int) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count, dtype=torch.float32)
    return shift * values / (1 + (shift - 1) * values)


def get_sampling_sigmas(
    num_inference_steps: int, shift: float = 5.0, training_shift: float = 1.0, num_train_timesteps: int = 1000,
    *, scheduler: str = "simple",
) -> torch.Tensor:
    """Return the sampling sigmas [steps + 1] (descending, the last value is 0) on the CPU in float32.

    Simple samples the shifted discrete table by index, matching ComfyUI.
    SGM uniform keeps continuous sigmas and exactly `num_inference_steps` steps.
    """
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    if scheduler not in ("simple", "sgm_uniform"):
        raise ValueError(f"Unsupported MOSS scheduler: {scheduler}")
    total = num_train_timesteps
    if scheduler == "sgm_uniform":
        # ComfyUI normal_scheduler(sgm=True) with ModelSamplingDiscreteFlow:
        # space steps + 1 timesteps uniformly, drop the last, shift, then append 0.
        # _sigma_grid includes the already-shifted lower endpoint used by that model.
        sigmas = _sigma_grid(num_inference_steps + 1, shift, total)[:-1]
        return torch.cat([sigmas, sigmas.new_zeros(1)])
    # ComfyUI simple_scheduler samples the shifted training table by index.
    # Rounding a separate linspace to the table uses a different schedule.
    times = torch.arange(1, total + 1, dtype=torch.float32) / total
    table = shift * times / (1 + (shift - 1) * times)
    stride = total / num_inference_steps
    sigmas = torch.stack([table[-(1 + int(i * stride))] for i in range(num_inference_steps)])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def euler_step(latents: torch.Tensor, model_pred: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    """One Euler step in float32: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v"""
    dt = (sigmas[step_index + 1] - sigmas[step_index]).to(torch.float32)
    return latents.float() + dt * model_pred.float()



@torch.no_grad()
def sample_image(model, vae, text, prompt, *, height=1024, width=1024, steps=None,
                 cfg_scale=None, negative_prompt="", sampler_name=None, scheduler=None,
                 distilled=False, device="cuda", dtype=None, step_callback=None,
                 phase_callback=None, seed=None, vram_policy=None):
    from PIL import Image
    if sampler_name not in (None, "euler") or scheduler not in (None, "simple", "sgm_uniform"):
        raise ValueError("MOSS sampling supports Euler with simple or sgm_uniform")
    if height % 16 or width % 16 or min(height, width) < 16:
        raise ValueError("MOSS image dimensions must be positive multiples of 16")
    if distilled:
        raise ValueError("MOSS-Image Base is not a distilled checkpoint")
    steps = 50 if steps is None else steps
    cfg = 3.0 if cfg_scale is None else cfg_scale
    dtype = dtype or model.dtype
    device = torch.device(device)
    if phase_callback:
        phase_callback("text")
    prompts = [prompt] + ([negative_prompt] if cfg != 1 else [])
    # The inference path can yield the DiT. During training, prompts are pre-cached.
    yielded = vram_policy is not None and not text.online_conditions_cached(prompts)
    if yielded:
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    try:
        conditions = text.encode_text_for_batch(prompts, device=device, dtype=dtype)
        text.release_model()
    finally:
        if yielded:
            model.to(device)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    latent = torch.randn((1, 128, height // 16, width // 16), generator=generator).to(device)
    sigmas = get_sampling_sigmas(
        steps, getattr(model, "sample_flow_shift", 5.0), scheduler=scheduler or "simple",
    ).to(device)
    skipped = getattr(model, "sample_uncond_skip_layers", (22,))
    training = model.training
    model.eval()
    def velocity(sigma, row, skip=()):
        context = conditions.context[row:row+1]
        mask = conditions.attention_mask[row:row+1]
        if getattr(model, "moss_comfy_forward", False):
            from .comfy_forward import comfy_forward
            return comfy_forward(model, latent.to(dtype), sigma.reshape(1),
                context[:, mask[0].bool()], skip_layers=skip).float()
        return model(latent.to(dtype), (sigma * 1000).reshape(1), context, mask, skip_layers=skip).float()
    try:
        if phase_callback:
            phase_callback("sample")
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            for index in range(len(sigmas) - 1):
                sigma = sigmas[index]
                denoised = latent - sigma * velocity(sigma, 0)
                if cfg != 1:
                    uncond = latent - sigma * velocity(sigma, 1, skipped)
                    denoised = uncond + (denoised - uncond) * cfg
                if step_callback:
                    step_callback(index, len(sigmas) - 1, unpatchify_latents(denoised).unsqueeze(2))
                pred = (latent - denoised) / sigma
                latent = euler_step(latent, pred, sigmas, index)
        if phase_callback:
            phase_callback("decode")
        pixels = vae.decode(unpatchify_latents(latent).unsqueeze(2))[0, :, 0].float().clamp(-1, 1)
        array = ((pixels.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
        return Image.fromarray(array)
    finally:
        model.train(training)
