"""Official MOSS Euler schedule, standard CFG and unconditional layer skipping.

Schedule from kohya-ss/sd-scripts 11d8cedd0c67b93a10f8b2d75f43554f6f12c794 (Apache-2.0).
"""
import torch
from modeling.moss_image.vae import unpatchify_latents

def _sigma_grid(count: int, shift: float, total: int) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count, dtype=torch.float32)
    return shift * values / (1 + (shift - 1) * values)


def get_sampling_sigmas(
    num_inference_steps: int, shift: float = 5.0, training_shift: float = 1.0, num_train_timesteps: int = 1000
) -> torch.Tensor:
    """Return the sampling sigmas [steps + 1] (descending, the last value is 0) on the CPU in float32.

    The number of effective steps can be smaller than `num_inference_steps` because the shifted grid is snapped to the
    training grid and duplicates are removed (same as the official scheduler).
    """
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    total = num_train_timesteps
    sigmas = _sigma_grid(num_inference_steps, shift, total)
    training = _sigma_grid(total, training_shift, total).flip(0)  # ascending
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
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
    if sampler_name not in (None, "euler") or scheduler not in (None, "simple"):
        raise ValueError("MOSS sampling supports Euler / simple")
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
    sigmas = get_sampling_sigmas(steps, getattr(model, "sample_flow_shift", 5.0)).to(device)
    skipped = getattr(model, "sample_uncond_skip_layers", (22,))
    training = model.training
    model.eval()
    try:
        if phase_callback:
            phase_callback("sample")
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            for index in range(len(sigmas) - 1):
                timestep = (sigmas[index] * 1000).reshape(1)
                pred = model(latent.to(dtype), timestep, conditions.context[:1], conditions.attention_mask[:1]).float()
                if cfg != 1:
                    uncond = model(latent.to(dtype), timestep, conditions.context[1:], conditions.attention_mask[1:], skip_layers=skipped).float()
                    pred = uncond + cfg * (pred - uncond)
                if step_callback:
                    step_callback(index, len(sigmas) - 1, unpatchify_latents(latent - sigmas[index] * pred).unsqueeze(2))
                latent = euler_step(latent, pred, sigmas, index)
        if phase_callback:
            phase_callback("decode")
        pixels = vae.decode(unpatchify_latents(latent).unsqueeze(2))[0, :, 0].float().clamp(-1, 1)
        array = ((pixels.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
        return Image.fromarray(array)
    finally:
        model.train(training)
