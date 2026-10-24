# Using your own checkpoints, VAE and text encoders

Settings → **Models** has three shared lists: **Checkpoints**, **VAE**, and **Text Encoders**. Official downloads and local files appear together, using the existing controls.

1. Set the global models root above the lists.
2. Place weights in its `diffusion_models`, `vae`, or `text_encoders` subfolder. These folders are scanned recursively without loading tensor data.
3. Select one recognized, available model per category. Split checkpoints appear once under a model name, with their combined size.
4. Reopen Settings after adding or moving files. Compatible family defaults remain remembered when switching the active model family.

| Weight | Supported local forms |
|---|---|
| Checkpoints | Anima/Krea2 single safetensors; MOSS single safetensors or complete official shards |
| VAE | Qwen-Image VAE for Anima/Krea2; FLUX.2 VAE for MOSS |
| Text Encoders | Qwen3/Qwen3-VL transformers directories; Qwen3.5 directory or compatible MOSS ConvRot single file |

Compatibility is detected from model contents, not filenames. Unknown formats remain visible but cannot be selected automatically. Existing registered local sources remain in the catalog. The original root folder and auto-sync setting remain available.

These defaults are used when resolving paths for a family. Existing YAML files are not rewritten by selecting a weight; with auto-sync enabled, workflows that refresh model paths follow the corresponding Settings defaults. Turn auto-sync off to keep explicit paths in a version configuration.

MOSS's VAE selection does not replace the shared Anima/Krea2 VAE. Quantization and layer offloading belong to the training configuration; see [MOSS-Image Base](moss-image.md).
