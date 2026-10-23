"""Original checkpoint module paths are retained in kohya-compatible LoRA keys."""
MOSS_PRESET = {
    "enable_conv": False, "target_module": ["MossImageBlock", "MossImageTextBlock"],
    "target_name": [], "exclude_name": [], "use_fnmatch": True,
    "lora_prefix": "lora_unet", "module_algo_map": {}, "name_algo_map": {},
}
