# JSON Caption Format Specification

AnimaLoraToolkit supports structured JSON tag files, which offer the following advantages over traditional TXT files:

- **Categorized shuffle**: appearance/tags/environment are each shuffled internally, preserving semantic structure
- **Fixed fields**: quality/character/series/artist always come first and are never shuffled
- **Easier to manage**: structured data is easier to batch-edit and version-control

## File structure

```
dataset/
├── image001.jpg
├── image001.json    # JSON file with the same name as the image
├── image002.png
├── image002.json
└── ...
```

## JSON schema

### Full format

```json
{
  "fixed": {
    "quality": "newest, safe",
    "series": "project name",
    "artist": "@artist name"
  },
  "character": {
    "name": "character name",
    "variant": ""
  },
  "from_path": {
    "appearance": ["blonde hair", "casual clothes"]
  },
  "ai_output": {
    "count": "1girl",
    "appearance": ["long hair", "blue eyes", "smile"],
    "tags": ["standing", "looking at viewer", "upper body"],
    "environment": ["outdoors", "sky", "sunlight"],
    "nl": "A cheerful girl stands under the bright sky."
  }
}
```

### Field reference

| Field | Type | Description |
|------|------|------|
| `fixed.quality` | string | Quality tag, always placed first |
| `fixed.series` | string | Work/project name |
| `fixed.artist` | string | Artist tag (must include `@`) |
| `character.name` | string | Character name |
| `character.variant` | string | Character variant (e.g. adult, alternate costume) |
| `from_path.appearance` | string[] | Appearance tags auto-extracted from the directory path |
| `ai_output.count` | string | Character count as recognized by the VLM |
| `ai_output.appearance` | string[] | Appearance features recognized by the VLM |
| `ai_output.tags` | string[] | Actions/expressions/composition recognized by the VLM |
| `ai_output.environment` | string[] | Environment/background recognized by the VLM |
| `ai_output.nl` | string | Natural language description |

## Simplified format

If you don't need the nested layering, you can use the simplified format instead:

```json
{
  "quality": "newest, safe",
  "count": "1girl",
  "character": "hatsune miku",
  "series": "vocaloid",
  "artist": "@wlop",
  "appearance": ["long hair", "blue hair", "twintails", "blue eyes"],
  "tags": ["singing", "microphone", "concert", "dynamic pose"],
  "environment": ["stage", "spotlight", "crowd", "night"],
  "nl": "Miku performs energetically on stage."
}
```

## Render order

The JSON is rendered into the final caption in the following order:

```
quality → count → character → series → artist → appearance → tags → environment. nl
```

**Example output**:
```
newest, safe, 1girl, hatsune miku, vocaloid, @wlop, long hair, blue hair, twintails, blue eyes, singing, microphone, concert, dynamic pose, stage, spotlight, crowd, night. Miku performs energetically on stage.
```

## Categorized shuffle

When `shuffle_caption: true` is enabled:

| Field | Shuffled? |
|------|----------|
| quality | ❌ Fixed |
| count | ❌ Fixed |
| character | ❌ Fixed |
| series | ❌ Fixed |
| artist | ❌ Fixed |
| appearance | ✅ Shuffled internally |
| tags | ✅ Shuffled internally |
| environment | ✅ Shuffled internally |
| nl | ❌ Always last |

**Shuffle example**:
```
# Original
appearance: ["long hair", "blue eyes", "school uniform"]
tags: ["smile", "standing", "looking at viewer"]

# After shuffling (example)
appearance: ["blue eyes", "school uniform", "long hair"]
tags: ["looking at viewer", "smile", "standing"]
```

## Working with batch_tag.py

`batch_tag.py` can automatically generate JSON files that conform to this format:

```bash
python batch_tag.py --input ./raw_images --output ./dataset --format json
```

The generated JSON includes:
- character/variant extracted from the directory structure
- count/appearance/tags/environment/nl tagged by the VLM
- the fixed fields from the config file

## Configuration example

```yaml
# config/my_training.yaml

data_dir: "./dataset"
prefer_json: true        # prefer JSON files
shuffle_caption: true    # enable categorized shuffle
keep_tokens: 0           # not needed in JSON mode
tag_dropout: 0.05         # optional: 5% random tag dropout
```

## Fallback mechanism

If a JSON file doesn't exist or fails to parse, it automatically falls back to the TXT file of the same name:

```
Priority: image001.json > image001.txt
```

## Migration guide

### Migrating from TXT to JSON

1. Re-tag with `batch_tag.py`, specifying `--format json`
2. Or convert manually:

```python
# txt_to_json.py
import json
from pathlib import Path

def convert(txt_path):
    tags = txt_path.read_text().strip()
    # simple parsing (assumes tags are already in order)
    parts = [t.strip() for t in tags.split(",")]
    
    json_data = {
        "quality": "newest, safe",
        "count": parts[2] if len(parts) > 2 else "1girl",
        "character": parts[3] if len(parts) > 3 else "",
        "series": parts[4] if len(parts) > 4 else "",
        "artist": parts[5] if len(parts) > 5 else "",
        "appearance": parts[6:10] if len(parts) > 6 else [],
        "tags": parts[10:15] if len(parts) > 10 else [],
        "environment": parts[15:] if len(parts) > 15 else [],
        "nl": ""
    }
    
    json_path = txt_path.with_suffix(".json")
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))

# batch convert
for txt in Path("./dataset").glob("*.txt"):
    convert(txt)
```

## Validation tool

Check whether a JSON file conforms to the format:

```python
# validate_json.py
import json
from pathlib import Path

REQUIRED_FIELDS = ["count"]
ARRAY_FIELDS = ["appearance", "tags", "environment"]

def validate(json_path):
    data = json.loads(json_path.read_text())
    
    # check required fields
    for field in REQUIRED_FIELDS:
        if field not in data and field not in data.get("ai_output", {}):
            print(f"Warning: {json_path} missing {field}")
    
    # check array fields
    for field in ARRAY_FIELDS:
        value = data.get(field) or data.get("ai_output", {}).get(field)
        if value and not isinstance(value, list):
            print(f"Warning: {json_path} {field} should be array")
    
    return True

for json_file in Path("./dataset").glob("*.json"):
    validate(json_file)
```
