# Using your own weights: base model / VAE / text encoder

In Settings → **Training Models**, every category of weights can be swapped out from the official download for a file you already have on disk:

| Weight | Form | Card |
|---|---|---|
| Main model (base model, transformer) | Single `.safetensors` | Anima main model / Krea 2 main model |
| VAE | Single `.safetensors` | Anima VAE (shared by both families) |
| Text encoder (CLIP / Qwen) | transformers **directory** (must contain `config.json`) | Qwen3-0.6B-Base (Anima) / Qwen3-VL-4B-Instruct (Krea 2) |

## How to add one

1. At the bottom of the relevant card, click **Choose file…** / **Choose directory…**, browse to where the weights live in the file picker that pops up, and select it.
2. The selected path appears as a "local" candidate row below the official variant; click its radio button to make it the default.
3. To stop using it, click **Unregister** — this only removes it from the list, **the file on disk is left untouched**. If the unregistered entry was the currently selected one, the selection automatically falls back to the official weights.

> **The path is relative to the machine running Studio.** In local mode that's your own computer; in cloud modes like Colab / Kaggle, the file picker browses the container's disk, so a local `D:\...` path doesn't exist there — upload the weights to the cloud drive (or the Drive mount point) first, then select them. The gray text next to the button tells you which case you're in.

If a file is deleted or moved, resolution automatically falls back to the official location instead of writing a dead path into the training config; the corresponding row on the card is marked "file not found" and its radio button is grayed out.

## Working mode (which model family a local base model belongs to)

The local base model row has a **mode** dropdown (Anima / Krea 2). This determines which family the weights are trained under, which in turn determines:

- the family defaults for the training config (sampler / scheduler / timestep sampling / caption-related toggles — see the capability matrix in `studio/domain/common.py`);
- the VAE and text encoder paths resolved alongside it;
- which fields are visible on the training page (capability bits that aren't supported are hidden and disabled).

Community fine-tuned weights are usually in the same family as their base: pick Anima for Anima fine-tunes, Krea 2 for Krea 2 fine-tunes. Changing the mode = moving that path from one family to another; if it was currently selected, it stays selected in the new family.

## When you need to touch the VAE / text encoder

Most people never need to — the official VAE (`qwen_image_vae`) and the official encoder are the ones actually used for training. Typical cases where you would switch:

- you've already downloaded the same weights elsewhere (ComfyUI / another trainer) and don't want to download them again → point directly to that copy;
- you have a quantized / trimmed encoder directory and want to save VRAM or disk space;
- you've done your own VAE fine-tune.

Studio only validates "file exists + extension matches" and "directory contains `config.json`" — it does **not** validate that the architecture matches. Pointing to an encoder that doesn't match the base model (e.g. pointing Qwen3-VL at an Anima setup) can still be added to the list, but training will error out when loading the weights — just switch back to the official one.

## Scope of effect

The selected value takes effect immediately for: the training config of any newly created version, test image generation, AI priors (regularization set generation), and post-training evaluation. Already-created versions also follow it when "auto-sync model paths" is on (the default in Settings); with that toggle off, each version keeps the path stored in its own yaml, which guarantees reproducibility.
