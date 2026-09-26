# Release Post Authoring Guide

Each markdown file in this directory is one post. Posts tagged `release` are the
source of truth for [`CHANGELOG.md`](../../CHANGELOG.md): `tools/bump_version.py`
renders the changelog from them (see ADR 0013). Never edit `CHANGELOG.md` by hand.

**This document covers file format / frontmatter / tooling. For what to write and
how, see [`CONTENT-GUIDE.md`](CONTENT-GUIDE.md).**

## One post = one file

- `<id>.md`, written in English.
- Recommended naming: `YYYY-MM-DD-slug`; release posts must be named
  `YYYY-MM-DD-v<version>.md` (the validator warns otherwise).

## Frontmatter fields

```yaml
---
date: 2026-06-28         # required, ISO date; used for ordering + display
tag: release             # required: release | notice | migration
title: v0.16.0 release   # required
pin: true                # optional, default false
version: "0.16.0"        # required for release posts
---
Body (markdown)
```

## Tags

| tag | meaning | when to use |
|---|---|---|
| `release` | update log | release notes for a version, one post per version |
| `notice` | announcement | general notice / heads-up |
| `migration` | migration | a behavior change that needs the user's attention or action |

Only `release` posts end up in the changelog.

## Tooling

```bash
python tools/bump_version.py validate          # check release post frontmatter
python tools/bump_version.py render-changelog  # rewrite CHANGELOG.md
python tools/bump_version.py bump              # sync version files + rewrite CHANGELOG.md
```

## Templates

Copy an existing post, e.g. `2026-07-19-v0.20.2.md` (`release`) or
`2026-06-28-url-root.md` (`migration`, pinned).
