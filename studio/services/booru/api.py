"""Shared Booru API layer (PP5).

`downloader.py` (PP2 single-tag bulk pull) and `reg_builder.py` (PP5 multi-tag
greedy iteration) need the same set of HTTP primitives: search, extract post
fields, fetch images. This module collects them into pure functions that don't
depend on DownloadOptions / RegBuildOptions, avoiding duplicate code paths.

Conventions:
- `search_posts` raises `requests.RequestException` on failure (caller must
  catch it); the return value is always a list (an empty list means "nothing
  on this page").
- `download_image` raises `RuntimeError` on failure (wrapping PIL / network
  errors); on success it returns the final on-disk path (including the
  PNG-renamed version, if applicable).
- No retry / sleep / cancel checks here — those are the caller's loop's
  responsibility.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

from ... import __version__


# UA identifying the app. Background (hotfix 0.5.x):
# - danbooru now sits behind Cloudflare bot-protection; no UA / the default
#   `python-requests/X.Y.Z` gets an immediate 403 -> the CF challenge page
#   ("Just a moment...");
# - spoofing a Chrome browser UA is actually more suspicious ("browser that
#   doesn't run JS" pattern) and also gets 403'd;
# - an app-name UA both passes the CF filter and satisfies danbooru's TOS
#   requirement to "use a descriptive User-Agent". gelbooru isn't as strict
#   but a descriptive UA is recommended there too.
_USER_AGENT_BASE = f"AnimaLoraStudio/{__version__}"


def _build_user_agent(username: str = "") -> str:
    """Build the UA. Prefer including (by username) — matches danbooru's
    recommended TOS format, and lets the backend attribute requests to a
    specific account, so when CF tightens up we're less likely to get swept
    in with the anonymous-UA crowd."""
    u = (username or "").strip()
    return f"{_USER_AGENT_BASE} (by {u})" if u else _USER_AGENT_BASE


def _api_headers(username: str = "") -> dict[str, str]:
    return {"User-Agent": _build_user_agent(username), "Accept": "application/json"}


def _download_headers(username: str = "") -> dict[str, str]:
    return {"User-Agent": _build_user_agent(username)}


def default_base_url(api_source: str) -> str:
    return (
        "https://gelbooru.com"
        if api_source == "gelbooru"
        else "https://danbooru.donmai.us"
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search_posts(
    api_source: str,
    tags_query: str,
    *,
    page: int = 1,
    limit: int = 100,
    user_id: str = "",
    api_key: str = "",
    username: str = "",
    base_url: Optional[str] = None,
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> list[dict[str, Any]]:
    """Generic booru search.

    `tags_query` is an already-assembled search string (multiple tags
    space-separated, including `-excluded` ones).
    Returns a list of posts (empty list = this page is empty / reached the end).
    """
    sess = session or requests
    url_base = base_url or default_base_url(api_source)
    if api_source == "gelbooru":
        params: dict[str, Any] = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "tags": tags_query,
            "pid": page - 1,
            "limit": min(limit, 100),
        }
        if api_key and user_id:
            params["api_key"] = api_key
            params["user_id"] = user_id
        url = f"{url_base}/index.php"
        auth = None
    else:
        params = {
            "tags": tags_query,
            "page": page,
            "limit": min(limit, 200),
        }
        url = f"{url_base}/posts.json"
        auth = (username, api_key) if username and api_key else None

    resp = sess.get(url, params=params, auth=auth, headers=_api_headers(username), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if api_source == "gelbooru":
        if isinstance(data, dict):
            posts = data.get("post")
            if isinstance(posts, dict):
                return [posts]
            if isinstance(posts, list):
                return posts
            if "@attributes" in data:
                return [data]
            return []
        if isinstance(data, list):
            return data
        return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# post field extraction
# ---------------------------------------------------------------------------


def post_fields(
    post: dict[str, Any], api_source: str
) -> tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Uniformly extract (post_id, file_url, file_ext, tags_str).

    gelbooru's fields can be at the top level or nested under `@attributes` —
    this handles both.
    """
    if api_source == "gelbooru" and "@attributes" in post:
        attrs = post["@attributes"]
        return (
            str(attrs.get("id")) if attrs.get("id") is not None else None,
            attrs.get("file_url"),
            str(attrs.get("file_ext", "jpg")),
            attrs.get("tags"),
        )
    return (
        str(post.get("id")) if post.get("id") is not None else None,
        post.get("file_url"),
        str(post.get("file_ext", "jpg")),
        post.get("tag_string") or post.get("tags"),
    )


def post_dimensions(
    post: dict[str, Any], api_source: str
) -> tuple[Optional[int], Optional[int]]:
    """Extract (width, height); also handles gelbooru's @attributes nesting."""
    if api_source == "gelbooru" and "@attributes" in post:
        attrs = post["@attributes"]
        w = attrs.get("width")
        h = attrs.get("height")
    else:
        w = post.get("image_width") or post.get("width")
        h = post.get("image_height") or post.get("height")
    try:
        return (int(w) if w is not None else None, int(h) if h is not None else None)
    except (TypeError, ValueError):
        return (None, None)


def post_tag_list(post: dict[str, Any], api_source: str) -> list[str]:
    """Extract the post's tag list (lowercased, space -> underscore, dedup order-preserving).

    gelbooru: the `tags` field is a space-separated string.
    danbooru: concatenated from `tag_string_general` / `_character` /
    `_copyright` / `_artist`.
    """
    raw: list[str] = []
    if api_source == "gelbooru":
        if "@attributes" in post:
            tags_str = post["@attributes"].get("tags", "")
        else:
            tags_str = post.get("tags", "")
        if tags_str:
            raw = [t.strip() for t in str(tags_str).split() if t.strip()]
    else:
        for cat in (
            "tag_string_general",
            "tag_string_character",
            "tag_string_copyright",
            "tag_string_artist",
        ):
            v = post.get(cat)
            if v:
                raw.extend(str(v).split())
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        norm = t.lower().replace(" ", "_")
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# ---------------------------------------------------------------------------
# image download
# ---------------------------------------------------------------------------


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "P") or "transparency" in img.info


def flatten_alpha(img: Image.Image) -> Image.Image:
    """Flatten the alpha channel onto a white background."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def download_image(
    url: str,
    save_path: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
    timeout: float = 60.0,
    referer: Optional[str] = None,
    session: Optional[requests.Session] = None,
    username: str = "",
) -> Path:
    """Download a single image to save_path (or its .png-renamed version);
    raises RuntimeError on failure.

    `username` is used to build the UA `(by username)`, consistent with the
    search path; downloads also go through Cloudflare, and a UA carrying an
    account identifier lowers the risk of being swept up with anonymous UAs.

    Returns the final on-disk path.
    """
    sess = session or requests
    headers = _download_headers(username)
    if referer:
        headers["Referer"] = referer
    resp = sess.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()
    raw = resp.content
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:
        raise RuntimeError(f"The image is corrupt or unreadable: {exc}") from exc

    final = save_path
    if convert_to_png and final.suffix.lower() != ".png":
        final = final.with_suffix(".png")
    if remove_alpha_channel and has_alpha(img):
        img = flatten_alpha(img)
    if final.suffix.lower() == ".png":
        out = (
            img.convert("RGBA")
            if has_alpha(img) and not remove_alpha_channel
            else img.convert("RGB")
        )
        out.save(final, "PNG", optimize=True)
    elif final.suffix.lower() in {".jpg", ".jpeg"}:
        img.convert("RGB").save(final, "JPEG", quality=95, optimize=True)
    elif final.suffix.lower() == ".webp":
        img.save(final, "WEBP", quality=95)
    else:
        final.write_bytes(raw)
    return final
