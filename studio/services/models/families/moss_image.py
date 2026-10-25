"""Official MOSS assets. VAE and text defaults are isolated from other families."""
from pathlib import Path
from .... import secrets
from ..paths import model_type_dir, models_root

REPO = "OpenMOSS-Team/MOSS-Image-Base"
TRANSFORMER_FILES = [f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors" for i in range(1, 4)]
TEXT_FILES = ["config.json", "model.safetensors.index.json", *[
    f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def asset_files(root, model_id):
    if model_id == "moss_image_main":
        return [(f"transformer/{f}", model_type_dir(root, "diffusion_models") / "moss-image-base" / f) for f in TRANSFORMER_FILES]
    if model_id == "moss_image_vae":
        return [("vae/diffusion_pytorch_model.safetensors", model_type_dir(root, "vae") / "flux2-vae.safetensors")]
    if model_id == "moss_image_text_encoder":
        return ([(f"text_encoder/{f}", model_type_dir(root, "text_encoders") / "moss-qwen35-9b" / f) for f in TEXT_FILES]
                + [(f"tokenizer/{f}", model_type_dir(root, "text_encoders") / "moss-qwen35-9b" / f) for f in TOKENIZER_FILES])
    raise ValueError(f"Unknown MOSS asset: {model_id}")


def _status(p):
    return {"exists": p.is_file(), "size": p.stat().st_size if p.is_file() else 0}


def transformer_path_for(sel=None):
    selected = sel or secrets.load().models.selected.get("moss_image", "bf16")
    if selected not in ("bf16", "latest"):
        return str(Path(selected).expanduser())
    return str(model_type_dir(models_root(), "diffusion_models") / "moss-image-base" / TRANSFORMER_FILES[0])


def default_paths_for_new_version(base_model=None):
    root = models_root()
    cfg = secrets.load().models
    return {"transformer_path": transformer_path_for(base_model),
            "vae_path": cfg.selected_vae_by_family.get("moss_image") or str(model_type_dir(root, "vae") / "flux2-vae.safetensors"),
            "text_encoder_path": cfg.selected_te.get("moss_image") or str(model_type_dir(root, "text_encoders") / "moss-qwen35-9b"),
            "t5_tokenizer_path": ""}


def text_encoder_presets(root):
    return [{"value": "", "label": "Qwen3.5-9B", "description": "MOSS text encoder (BF16)",
             "download_id": "moss_image_text_encoder", "status_key": "moss_image_text_encoder",
             "path": str(model_type_dir(root, "text_encoders") / "moss-qwen35-9b"),
             "files": [{"name": p.name, **_status(p)} for _, p in asset_files(root, "moss_image_text_encoder")]}]


def catalog_sections(root, models_cfg):
    files = [p for _, p in asset_files(root, "moss_image_main")]
    return {"moss_image_main": {"id": "moss_image_main", "name": "MOSS-Image Base", "description": "5B BF16 transformer",
        "repo": REPO, "latest": "bf16", "selected": models_cfg.selected.get("moss_image") or "bf16",
        "license": "Apache-2.0", "license_url": f"https://huggingface.co/{REPO}/blob/main/LICENSE",
        "variants": [{"variant": "bf16", "purpose": "training", "target_path": str(files[0]),
            "exists": all(p.is_file() for p in files), "size": sum(_status(p)["size"] for p in files)}],
        "custom": [{"path": p, "name": Path(p).name, **_status(Path(p))} for p in models_cfg.custom.get("moss_image", [])]}}


class _MossAssets:
    family_id = "moss_image"
    display_name = "MOSS-Image Base"
    latest = "bf16"
    default_paths_for_new_version = staticmethod(default_paths_for_new_version)
    transformer_path_for = staticmethod(transformer_path_for)
    selected_variant = staticmethod(lambda: secrets.load().models.selected.get("moss_image") or "bf16")
    catalog_sections = staticmethod(catalog_sections)
    text_encoder_presets = staticmethod(text_encoder_presets)
    is_distilled_path = staticmethod(lambda path: False)


ASSETS = _MossAssets()
