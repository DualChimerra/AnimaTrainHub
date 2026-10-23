"""MOSS-Image Base family specification."""
from ..spec import ModelSpec, TextSpec, SamplingDefaults, ConstantShift, LoraOutputSpec
from ..latent_spaces import FLUX2_F8C32
from studio.domain.common import FAMILY_CAPABILITIES, FAMILY_CONFIG_DEFAULTS, FAMILY_SAMPLING

MOSS_SPEC = ModelSpec(
    family_id="moss_image", display_name="MOSS-Image Base", objective="rectified_flow",
    latent=FLUX2_F8C32,
    text=TextSpec(strategy="cached_varlen", max_seq_len=4096, fingerprint="moss-qwen35-9b-last4096-v1"),
    sampling=SamplingDefaults(samplers=FAMILY_SAMPLING["moss_image"]["samplers"],
        schedulers=FAMILY_SAMPLING["moss_image"]["schedulers"], default_sampler="euler",
        default_scheduler="simple", default_steps=50, default_cfg=3.0, shift_policy=ConstantShift(5.0)),
    capabilities=FAMILY_CAPABILITIES["moss_image"],
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="moss_image"),
    config_defaults=FAMILY_CONFIG_DEFAULTS["moss_image"],
)
