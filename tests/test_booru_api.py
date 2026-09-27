"""services/booru_api.py -- regression tests for search_posts headers / routing.

Hotfix background: after danbooru put up Cloudflare, a missing User-Agent
(requests' default `python-requests/X.Y.Z`) gets a 403 challenge page straight
from CF. This pins down the two key invariants:
- requests must carry a recognizable UA (containing 'AnimaLoraStudio')
- Accept: application/json makes middleware routing more deterministic

No real HTTP is sent: a fake session intercepts and verifies sess.get's arguments.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from studio.services.booru import api as booru_api


def _fake_session(json_payload: Any = None) -> MagicMock:
    sess = MagicMock()
    resp = MagicMock()
    resp.json.return_value = json_payload if json_payload is not None else []
    resp.raise_for_status.return_value = None
    sess.get.return_value = resp
    return sess


def test_search_posts_sends_app_user_agent_for_danbooru() -> None:
    sess = _fake_session([{"id": 1}])
    booru_api.search_posts("danbooru", "1girl", session=sess)
    _, kwargs = sess.get.call_args
    headers = kwargs.get("headers") or {}
    assert "User-Agent" in headers
    assert "AnimaLoraStudio" in headers["User-Agent"]
    assert headers.get("Accept") == "application/json"


def test_search_posts_ua_includes_username_when_provided() -> None:
    """When searching with a username, the UA should be 'AnimaLoraStudio/X (by username)' --
    the format danbooru's TOS recommends, so CF can allowlist by account instead of blocking anonymously."""
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "1girl", username="alice", session=sess)
    ua = sess.get.call_args.kwargs["headers"]["User-Agent"]
    assert "AnimaLoraStudio" in ua
    assert "(by alice)" in ua


def test_search_posts_ua_falls_back_when_no_username() -> None:
    """When no username is passed, the UA should not carry empty parentheses."""
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "1girl", session=sess)
    ua = sess.get.call_args.kwargs["headers"]["User-Agent"]
    assert "(by" not in ua
    assert "AnimaLoraStudio/" in ua


def test_search_posts_sends_app_user_agent_for_gelbooru() -> None:
    """gelbooru is less strict, but should still carry a UA for polite usage."""
    sess = _fake_session({"post": [{"id": 2}]})
    booru_api.search_posts("gelbooru", "1girl", session=sess)
    _, kwargs = sess.get.call_args
    headers = kwargs.get("headers") or {}
    assert "AnimaLoraStudio" in headers["User-Agent"]


def test_build_user_agent_strips_whitespace() -> None:
    """When username has leading/trailing spaces or is all spaces, it should not generate an empty UA segment like '(by   )'."""
    assert "(by" not in booru_api._build_user_agent("   ")
    assert booru_api._build_user_agent("  alice  ") == \
        f"{booru_api._USER_AGENT_BASE} (by alice)"


def test_search_posts_user_agent_does_not_impersonate_browser() -> None:
    """Regression: a Chrome browser UA was once used but Cloudflare flagged it as
    "a browser that doesn't run JS" and returned 403. The UA must clearly identify
    itself as AnimaLoraStudio."""
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "x", session=sess)
    ua = sess.get.call_args.kwargs["headers"]["User-Agent"]
    assert "Mozilla" not in ua
    assert "Chrome" not in ua


def test_search_posts_routes_danbooru_correctly() -> None:
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "1girl rating:safe", page=2, limit=50, session=sess)
    args, kwargs = sess.get.call_args
    url = args[0]
    params = kwargs["params"]
    assert url.endswith("/posts.json")
    assert params["tags"] == "1girl rating:safe"
    assert params["page"] == 2
    assert params["limit"] == 50


def test_search_posts_routes_gelbooru_correctly() -> None:
    sess = _fake_session({"post": []})
    booru_api.search_posts("gelbooru", "tag1", page=3, limit=100, session=sess)
    args, kwargs = sess.get.call_args
    url = args[0]
    params = kwargs["params"]
    assert url.endswith("/index.php")
    assert params["page"] == "dapi"
    assert params["pid"] == 2  # page-1 for gelbooru
    assert params["limit"] == 100


def test_search_posts_passes_basic_auth_for_danbooru_when_provided() -> None:
    sess = _fake_session([])
    booru_api.search_posts(
        "danbooru", "x",
        username="me", api_key="secret",
        session=sess,
    )
    _, kwargs = sess.get.call_args
    assert kwargs["auth"] == ("me", "secret")


def test_search_posts_no_auth_when_username_missing() -> None:
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "x", api_key="secret", session=sess)
    assert sess.get.call_args.kwargs["auth"] is None


def test_download_image_ua_includes_username(tmp_path) -> None:
    """The download path also goes through CF; the UA must likewise carry the username for the account allowlist to work."""
    from io import BytesIO
    from PIL import Image as _PILImage
    buf = BytesIO()
    _PILImage.new("RGB", (4, 4), (255, 0, 0)).save(buf, "PNG")
    png_bytes = buf.getvalue()

    sess = MagicMock()
    resp = MagicMock()
    resp.content = png_bytes
    resp.raise_for_status.return_value = None
    sess.get.return_value = resp

    booru_api.download_image(
        "https://cdn.donmai.us/x.png",
        tmp_path / "x.png",
        convert_to_png=False,
        remove_alpha_channel=False,
        username="alice",
        session=sess,
    )
    headers = sess.get.call_args.kwargs["headers"]
    assert "(by alice)" in headers["User-Agent"]
