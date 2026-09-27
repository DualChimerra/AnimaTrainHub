# Announcement Content Guide

**Applies to**: what to write and how to write it for every post under `docs/announcements/` —
`release` (update log), `notice` (announcement), `migration` (behavior change / needs attention),
and future types.
**File format / frontmatter / tags / tooling** are covered in [`README.md`](README.md); this
document only covers content and tone.

> This guide draws on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
> [Apple HIG · Writing](https://developer.apple.com/design/human-interface-guidelines/writing),
> and industry release-notes practice ([ProductPlan](https://www.productplan.com/learn/release-notes-best-practices),
> [Appcues](https://www.appcues.com/blog/release-notes-examples)).

---

## First principles (apply to every announcement)

1. **Write for the user, not the engineer.** Describe the change in the user
   experience, not the implementation. When users skip release notes, it's usually
   not that they don't care — it's that the notes were written from an engineering
   point of view.
2. **Scannable: the first sentence of each entry answers "what changed + does it
   affect me".** Users scanning an announcement only want these two answers, so
   put them in the first sentence.
3. **Plain language, minimal jargon.** Avoid unexplained technical terms, internal
   code names, or implementation details — write as if explaining to a
   non-programmer friend.
4. **Only list differences the user can perceive versus the previous version.** A
   bug introduced and fixed within the same dev cycle doesn't count. A regression
   guard is not a "fix." Pure internal refactors or one-off workarounds don't
   belong here. One entry = one user-perceivable unit of change.
5. **Be concise.** Check every word for necessity; trim what you can. Active voice,
   present tense.

## What to write for each announcement type

### `release` — update log (one post per version)
- Title: `en "v0.16.0 release"`; `date` uses ISO `YYYY-MM-DD`; the newest version
  appears first in the announcement panel.
- Body grouped by change type (Keep a Changelog's categories): **Added / Changed /
  Improved / Fixed / Deprecated / Removed / Security**. List only the groups that
  actually have entries this version.
- Under each group, each bullet: **bold the first sentence** (feature/action name,
  ending with the PR number `(#NN)`) so it's scannable at a glance; immediately
  follow with a detail sentence **in the same paragraph** (no blank line → renders
  as "**first sentence** detail" as one paragraph), then a blank line and `  - `
  for sub-bullets if needed.
- **Deprecations and breaking changes must be listed prominently** so users can
  plan their upgrade — these are the best candidates for a dedicated `migration`
  post (see below).

### `notice` — general announcement
- Notices/heads-up items (e.g. a feature launch, maintenance notes, community
  info). One sentence stating what it is + what it means for the user.
- No need to group by kind — one post per piece of information.

### `migration` — behavior change / needs attention or action (most important)
Trigger: a user-perceivable behavior changed, a default changed, an entry
point/address/path changed, or the user needs to do something.

- **Open with one sentence stating what changed** — don't bury it in details.
- **Always give action guidance** — this is what distinguishes a migration post
  from a regular announcement:
  - No action needed → say explicitly "**takes effect automatically, no action
    needed**."
  - Action needed → give concrete, actionable steps, e.g. "go to **Settings → X →
    Y** to change it" or "old bookmarks will redirect automatically; consider
    updating them to…".
- Set `pin: true` for important ones so they stay at the top.

## Tone / person / tense / length
- Tone: plain, friendly, with some character but without sacrificing information
  density; not overly casual or meme-heavy.
- Person: address the user as "you"; describe changes in active voice, present
  tense ("you can now…", not "we refactored…").
- Length: one line if one line says it; put details in a sub-section below rather
  than a wall of text.

## Don't write
- Internal refactors / module moves / dependency bumps that users can't perceive
  (unless they cause a perceivable behavior change).
- "Fixed a regression" — a regression guard isn't a user-facing fix (the user
  never saw that bug).
- One-off workarounds / debugging details / commit-log-style listings (noise).
- Hype ("completely solved", "perfect support") or unexplained abbreviations and
  code names.
- Technical comparisons ("aligned with industry practice", "modeled on X's
  implementation") — these belong in the PR description, not the announcement.

## Structure / markdown conventions
- Use real markdown: `### group heading`, `- ` lists, at most **one level** of
  nested sub-bullets when needed (indentation aligned); avoid deeper nesting.
- Give links as full URLs or markdown links; wrap code/paths/field names in
  `` ` ``.
- Dates in ISO `YYYY-MM-DD`; avoid regional ambiguity.
- Don't bold whole paragraphs or use all caps for emphasis — let structure (pinning,
  first sentence, grouping) carry emphasis, not visual noise.

## Localization
- Write each language natively, not as a machine translation of the other: English
  copy should read like release notes written by a native English speaker (Apple's
  concise, friendly style), not translated English.
- Title conventions are listed per type above; body content should be
  informationally equivalent across languages without needing to match word for
  word.

## References
- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
- [Apple HIG · Writing](https://developer.apple.com/design/human-interface-guidelines/writing)
- [ProductPlan · Release notes best practices](https://www.productplan.com/learn/release-notes-best-practices)
- [Appcues · 13 release notes examples](https://www.appcues.com/blog/release-notes-examples)
