# families/ -- model family registry (plugin registry #8, architecture-level)

Authoritative design reference: `docs/design/multi-model/04-synthesis.md` (frozen
interface surface in 03 SS4.1 + 04 SS3).

## A family's three homes

| Layer | Location | Contents |
|---|---|---|
| Structure definition | `modeling/<fam>/` | DiT / wrapper layers (depends only on torch/einops) |
| Behavior adapter | `runtime/training/families/<fam>/` | ModelFamily implementation: loader / forward / preset / sampling / text_encoding |
| Asset manifest | `studio/services/models/families/<fam>.py` | weight repo / download target / default paths (landed in PR-4) |

The family name string (`anima` / `krea2`) is the single join key running through all three layers.

## Boundary discipline

- The shared loop only consumes `(latents, noise, t, pred, target, loss, mask, loss_weight)`;
  any model knowledge outside of that (text encoding, pad_mask, checkpoint
  unpacking, sampling stack) belongs inside the family.
- **Shared code must never branch on `if family == "..."`** -- always check
  `spec.capabilities` or a spec field instead.
- The cache fingerprint is the latent-space identity (`wan21-f8c16`), not the
  family name: families that share a latent space automatically share the cache.
- **Family-agnostic shared facts don't live under any one family's name**:
  latent spaces are defined in `latent_spaces.py` (e.g. `WAN21_F8C16`, with
  both family specs referencing the same instance); shared studio-side assets
  (where the Qwen-Image VAE lives) go in `services/models/paths.py`. Rule of
  thumb: an external fact used by multiple families is placed in a neutral
  layer -- never copied, and never imported from one family into another.
- Krea2's `text_encoder_cache=true` pre-caches and releases Qwen3-VL before
  loading the DiT; when disabled, it never reads/writes the text sidecar at
  all, and TE + DiT stay resident, for high-VRAM/low-disk cloud setups that
  encode batch by batch.
- Evolution: new method parameters are always keyword-only with a default;
  `**kwargs` is forbidden.

## Steps to add a 3rd family

1. Put the structure definition in `modeling/<fam>/` (module naming aligned
   with ComfyUI's internal naming -- kohya key names encode the module path).
2. Create `<fam>/` in this directory: `__init__.py` (SPEC) + `family.py` +
   `loader.py` + `preset.py` + `sampling.py` (+ `text_encoding.py` for
   families with a text cache). If the latent space matches an existing
   family, have the SPEC reference the existing instance in
   `latent_spaces.py` directly; for a new space, add one there.
3. Register: one line each in `families/__init__.py`'s `_register(SPEC)` and
   the `get_family()` branch.
4. Schema: add the value to the `model_family` Literal + capability gating via
   `show_when` (PR-3 mechanism).
5. Studio: add the manifest to `FAMILY_ASSETS` (PR-4 mechanism).
6. Tests: `tests/test_families_<fam>_*.py` (spec constants / preset / flat key samples).

The shared loop (masked loss / InfoNoise / losses / optimizer / eval / pause-resume) needs zero changes.
