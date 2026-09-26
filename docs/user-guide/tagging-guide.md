# Anima LoRA Tagging Guide

> Best practices for tagging, based on the Anima Model Card and the official ComfyUI implementation

## Core rules

### 1. Tag format: use spaces, not underscores

Per the official examples in the Anima Model Card and the ComfyUI implementation:

```
✅ Correct: oomuro sakurako, yuru yuri, brown hair, long hair
❌ Wrong: oomuro_sakurako, yuru_yuri, brown_hair, long_hair
```

**Why**: ComfyUI's tokenizer passes text straight to the Qwen2Tokenizer/T5Tokenizer without converting underscores. Anima's training data uses space-separated tags.

### 2. Tag order (official recommendation)

```
quality/safety → count → character → series → artist → appearance → tags → environment. natural language description
```

| Position | Field | Example |
|------|------|------|
| 1 | quality | `newest, safe` |
| 2 | count | `1girl`, `2boys`, `no humans` |
| 3 | character | `hatsune miku` |
| 4 | series | `vocaloid` |
| 5 | artist | `@wlop` |
| 6 | appearance | `long hair, blue eyes, twintails` |
| 7 | tags | `smile, standing, looking at viewer` |
| 8 | environment | `concert stage, spotlight, crowd` |
| 9 | nl | `.` followed by a natural language description |

### 3. Artist tags must have the `@` prefix

```
✅ Correct: @wlop, @sakimichan, @torino aqua
❌ Wrong: wlop, sakimichan, torino aqua
```

**Important**: an artist tag without the `@` prefix barely has any effect!

### 4. Quality tag recommendations

When training a LoRA, **avoid** complex quality tags — keep it to just:

```
newest, safe
```

This lets the LoRA focus on learning style and character, while quality tags get added by the user at inference time.

**Full quality tag vocabulary** (for inference):
- Human rating: `masterpiece` > `best quality` > `good quality` > `normal quality`
- Aesthetic score: `score_9` > `score_8` > ... > `score_1`
- Year: `newest`, `recent`, `mid`, `early`, `old`, or `year 2024`
- Safety: `safe`, `sensitive`, `nsfw`, `explicit`

---

## Character variant naming

Use a **space + parentheses** to denote variants:

| Variant type | Tag format |
|----------|----------|
| Base character | `hatsune miku` |
| Specific outfit | `hatsune miku (racing)` |
| Age variant | `hatsune miku (adult)` |
| Alternate timeline/form | `hatsune miku (append)` |

---

## Fixed fields vs. dynamic fields

### Fixed fields (auto-filled based on project/directory)

For LoRA training on a specific project, it's recommended to fix the following fields:

| Field | Description | Example |
|------|------|------|
| quality | Uniform quality tag | `newest, safe` |
| series | Work/project name | `my project` |
| artist | Artist/style tag | `@my artist` |
| character | Character name (can be mapped from directory) | `character a` |

### Dynamic fields (VLM-tagged)

The following fields need to be generated dynamically for each image:

| Field | Description |
|------|------|
| count | Character count (`1girl`, `2boys`, `no humans`) |
| appearance | Character appearance (hairstyle, hair color, eye color, clothing, accessories) |
| tags | Actions, expressions, composition, held items |
| environment | Background, scene, lighting, atmosphere |
| nl | 1-2 sentence natural language description (placed last, separated by a period) |

---

## VLM tagging

### System prompt template

```
You are an anime image tagging expert. Output ONLY valid JSON.

JSON fields (tag fields are arrays of lowercase strings):
1. count: string - Character count ("1girl", "2boys", "1girl, 1boy", "no humans")
2. appearance: string[] - Visual features (hair color, eye color, hairstyle, clothing, accessories)
3. tags: string[] - Actions, expressions, poses, composition, objects
4. environment: string[] - Background, location, lighting, atmosphere
5. nl: string - One sentence natural language description

Rules:
- Use lowercase English booru-style tags
- Each tag is a separate array element
- Only describe what is clearly visible
- Be detailed but don't repeat tags
- Output ONLY the JSON object, no markdown or explanation

Example:
{"count": "1girl", "appearance": ["long hair", "blue eyes", "school uniform"], "tags": ["smile", "standing", "looking at viewer"], "environment": ["classroom", "window", "sunlight"], "nl": "A cheerful girl stands by the window in a sunny classroom."}
```

### API call parameters (Gemini recommended)

```python
{
    "generationConfig": {
        "temperature": 0.2,
        "topP": 0.8,
        "maxOutputTokens": 512,
        "thinkingConfig": {
            "thinkingBudget": 128
        }
    }
}
```

**Key parameters**:
- `thinkingBudget: 128` - limits thinking tokens, keeps the output clean
- **Don't** add words like `SPECIAL INSTRUCTION` or `Danbooru` to the prompt — this may trigger safety filters

---

## Automatic directory-structure mapping

### Creating a character map

Create a character-mapping dictionary for your project:

```python
# Example: character directory name → English tag
CHAR_MAP = {
    "CharacterA": "character a",
    "CharacterA-variant": "character a (variant)",
    "CharacterB": "character b",
    # ... add your characters
}
```

### Variant/outfit mapping

Subfolder names can be automatically parsed and turned into tags:

| Type | Example directory name | English tag | Added to |
|------|-----------|----------|--------|
| **Age** | `adult` | `(adult)` | character name suffix |
| **Hair color** | `blonde` | `blonde hair` | appearance |
| **Outfit** | `kimono` | `kimono` | appearance |
| **Outfit** | `swimsuit` | `swimsuit` | appearance |
| **Outfit** | `school uniform` | `school uniform` | appearance |
| **State** | `fighting` | `fighting stance` | tags |

**Example path parsing**:
- `CharacterA/blonde-casual/xxx.png` → `character a, blonde hair, casual clothes`
- `CharacterB/kimono/xxx.png` → `character b, kimono`

---

## Final caption example

**Input image**: `character/CharacterA/kimono/001.png`

**VLM output (JSON format)**:
```json
{
  "count": "1girl",
  "appearance": ["long hair", "black hair", "red eyes", "hair ornament"],
  "tags": ["standing", "smile", "looking at viewer", "upper body"],
  "environment": ["indoors", "traditional room", "soft lighting"],
  "nl": "A graceful girl in traditional attire smiles warmly in a serene room."
}
```

**Fixed fields**:
```python
FIXED = {
    "quality": "newest, safe",
    "series": "my project",
    "artist": "@my artist",
}
```

**Auto-added from path**: `kimono` (from the subfolder `kimono`)

**Final caption**:
```
newest, safe, 1girl, character a, my project, @my artist, long hair, black hair, red eyes, hair ornament, kimono, standing, smile, looking at viewer, upper body, indoors, traditional room, soft lighting. A graceful girl in traditional attire smiles warmly in a serene room.
```

---

## Training parameter recommendations

### TXT mode

```yaml
shuffle_caption: true   # shuffle tags
keep_tokens: 6           # protect the first 6 tags from shuffling
                        # newest, safe, 1girl, character, series, artist
```

### JSON mode (recommended)

```yaml
prefer_json: true       # use JSON files
shuffle_caption: true   # shuffle within each category (appearance/tags/environment)
keep_tokens: 0           # fixed fields are automatically placed first in JSON mode
```

---

## FAQ

### Q: Why doesn't my artist tag have any effect?

A: Check whether you included the `@` prefix. `@wlop` works, `wlop` doesn't.

### Q: Should character names use underscores or spaces?

A: **Spaces**. Anima's training data uses spaces; underscores get treated as literal characters.

### Q: Do I need a lot of quality tags?

A: Only use `newest, safe` for training. Add things like `masterpiece, best quality` at inference time.

### Q: Where does the natural language description go?

A: At the end, separated by a `.` (period): `..., environment tags. Natural language description here.`

---

## References

- [Anima Model Card](https://huggingface.co/circlestone-labs/Anima)
- [ComfyUI-AnimaTool Prompt Guide](https://github.com/Moeblack/ComfyUI-AnimaTool/wiki/Prompt-Guide)
- [ComfyUI source - comfy/text_encoders/anima.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/anima.py)
