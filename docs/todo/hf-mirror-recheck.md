# hf-mirror.com recheck list (0.8.2 hotfix leftover)

**Created** 2026-05-17
**Triggering version** 0.8.2 hotfix
**Current status** 🔴 hf-mirror.com downloads fail across every tested `huggingface_hub` version (0.25 / 0.30 / 0.34 / 1.14). The preset has been hidden from the UI, and the default endpoint has been switched back to the official HF source.

---

## Symptom

When the UI / CLI goes through `huggingface_hub.hf_hub_download(endpoint="https://hf-mirror.com")`, it fails every time:

```
huggingface_hub.errors.FileMetadataError:
  Distant resource does not seem to be on huggingface.co.
  It is possible that a configuration issue prevents you from downloading
  resources from https://huggingface.co. Please check your firewall and proxy
  settings and make sure your SSL certificates are updated.
```

Internal library logic: the HEAD response is missing `X-Repo-Commit` → `commit_hash is None` → the above error is raised (see `huggingface_hub/file_download.py:_get_metadata_or_catch_error`).

## Root cause (confirmed as of 2026-05-17)

- **Not an upstream regression**: 0.25.2 (the requests-based era) fails the same way, so it's **not** related to the switch to httpx or cross-domain redirects not being followed.
- **It's a server-side change on hf-mirror's end**: curl following redirects gets the bytes just fine (200, byte count matches), but `huggingface_hub` can't read `commit_hash`. This means some hop in the response chain has headers that don't match what the hub expects — suspected to be that hf-mirror's 308 redirect back to huggingface.co doesn't carry the headers the hub needs for validation.
- Upstream PR [`huggingface/huggingface_hub#4071`](https://github.com/huggingface/huggingface_hub/pull/4071) (filed by a JFrog engineer, 2026-04) adds opt-in cross-domain HEAD redirect following, but it is **not merged**. Even if merged, it's opt-in (requires the env var `HF_HUB_ALLOWED_HEAD_REDIRECT_HOSTS`), so simply upgrading hub **will not automatically fix** hf-mirror.

## Recheck command (run this on every recheck)

Run this in the main venv:

```powershell
$py = "G:\AnimaLoraStudio\venv\Scripts\python.exe"
$code = @"
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
for ep in ['https://hf-mirror.com', 'https://huggingface.co']:
    t0 = time.time()
    try:
        p = hf_hub_download(repo_id='google/t5-v1_1-xxl',
                            filename='tokenizer_config.json',
                            endpoint=ep, force_download=True)
        print(f'[OK  {(time.time()-t0)*1000:.0f}ms] {ep}  size={Path(p).stat().st_size}')
    except Exception as e:
        print(f'[ERR {(time.time()-t0)*1000:.0f}ms] {ep}  {type(e).__name__}: {str(e)[:120]}')
"@
& $py -c $code
```

**Verdict**: hf-mirror's line prints `[OK ...]` with byte count = 1857 → the service is back; still `[ERR ...]` → keep waiting.

Attached curl comparison probe (to confirm whether it's a hub client issue or the mirror service is actually down):

```powershell
foreach ($u in @(
  'https://hf-mirror.com/google/t5-v1_1-xxl/resolve/main/tokenizer_config.json',
  'https://huggingface.co/google/t5-v1_1-xxl/resolve/main/tokenizer_config.json'
)) {
  & curl.exe -sSL -o NUL -w "HTTP %{http_code} | size=%{size_download} | redirects=%{num_redirects}`n" $u
}
```

Both sides showing `HTTP 200 | size=1857` = the mirror service itself is alive, and the problem is in the metadata compatibility between the hub client and the mirror.

## What to restore once it's back

If the probe above shows hf-mirror as `[OK]` too, roll back the hotfix following this list:

1. **`studio/secrets.py`** — `HuggingFaceConfig.endpoint` default value, depending on the situation:
   - if domestic (China-based) users are still the majority → change back to `"https://hf-mirror.com"`
   - if the project's focus has shifted overseas (check README / monitoring) → keep `""`
2. **`studio/web/src/pages/tools/Settings.tsx`** — add back `{ value: 'https://hf-mirror.com', label: 'hf-mirror.com', hint: 'Recommended for users in China (community-maintained mirror)' }` to `HF_ENDPOINT_PRESETS`, and sync the hint at the top with the default value
3. **`studio/web/src/pages/tools/Settings.tsx`** — restore the `helpTooltip` copy along the lines of "hf-mirror recommended for China, official source recommended overseas"
4. **`studio/web/src/pages/tools/Settings.test.tsx`** — sync `initialServerState.huggingface.endpoint` mock with whatever the default value changes to
5. **`README.md`** — restore the hf-mirror description in the "2. Downloading models in Studio" section (see git log for the original text), remove the 0.8.2 hotfix warning block
6. **`docs/architecture/studio-pipeline.md`** — change the `secrets.jsonc` example endpoint back to `"https://hf-mirror.com"` (if the default value is restored), remove the hotfix comment
7. **`tools/download_models.py`** — restore the epilog text saying "hf-mirror.com is the default on first install," remove the 0.8.2 note
8. **This file** — move to `docs/todo/archive/` or delete it outright, add a CHANGELOG entry describing the restoration

## Recommended recheck cadence

- **Every 2 weeks** (2026-05-31 / 2026-06-14 / 2026-06-28 ...) run the recheck command above
- **Triggered recheck**: when upstream PR #4071 is merged and released (subscribe to GitHub notifications), separately evaluate whether the `HF_HUB_ALLOWED_HEAD_REDIRECT_HOSTS` env var can fix things in place
- **If still broken after 3 months** (after 2026-08-17), seriously consider whether hf-mirror should be deprecated long-term — even if the preset can be restored, it should no longer be listed as a recommended source

## Upstream / community tracking points

- [`huggingface/huggingface_hub` PR #4071](https://github.com/huggingface/huggingface_hub/pull/4071) — cross-domain HEAD redirect opt-in (open, 2026-04)
- [hf-mirror.com](https://hf-mirror.com/) — if there's an announcement or maintenance notice, it will be posted on the homepage
- No GitHub / email channel found for the hf-mirror maintainers; can only wait passively.
