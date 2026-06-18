"""Integration test for browser download functionality.

Tests both URL-based and click-based downloads using a tiny local HTTP
server that serves a test file with Content-Disposition headers.

Run:  python -m pytest tests/test_browser_download.py -v -s
"""

from __future__ import annotations

import asyncio
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread

import pytest

from arion_agent.environments.browser.config import BrowserConfig
from arion_agent.environments.browser.session import BrowserSession


SAMPLE_CONTENT = b"Hello from test download file"
SAMPLE_FILENAME = "test_document.txt"

HTML_PAGE = f"""\
<html><body>
<a id="dl-link" href="/download">Download file</a>
</body></html>
"""


class _Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/download":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{SAMPLE_FILENAME}"',
            )
            self.send_header("Content-Length", str(len(SAMPLE_CONTENT)))
            self.end_headers()
            self.wfile.write(SAMPLE_CONTENT)
        elif self.path == "/":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture()
def save_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
async def session():
    config = BrowserConfig(headless=True, stealth=False, humanize=False)
    s = BrowserSession(config)
    yield s
    if s.is_open:
        await s.close()


@pytest.mark.asyncio
async def test_download_url_based(server, save_dir, session):
    """Download via direct URL — non-blocking, then poll status."""
    url = f"{server}/download"

    result = await session.download(url, save_dir, tab="default")
    assert "dl_1" in result
    assert "started" in result.lower()

    for _ in range(30):
        await asyncio.sleep(0.5)
        listing = session.download_list()
        if "complete" in listing:
            break
    else:
        pytest.fail(f"Download did not complete in time.\n{session.download_list()}")

    saved = save_dir / SAMPLE_FILENAME
    assert saved.exists(), f"Expected {saved}, dir contains: {list(save_dir.iterdir())}"
    assert saved.read_bytes() == SAMPLE_CONTENT


@pytest.mark.asyncio
async def test_download_click_based(server, save_dir, session):
    """Download via clicking a link on the page."""
    await session.navigate(f"{server}/", tab="default")

    result = await session.download("#dl-link", save_dir, tab="default")
    assert "dl_" in result
    assert "started" in result.lower()

    for _ in range(30):
        await asyncio.sleep(0.5)
        listing = session.download_list()
        if "complete" in listing:
            break
    else:
        pytest.fail(f"Download did not complete in time.\n{session.download_list()}")

    saved = save_dir / SAMPLE_FILENAME
    assert saved.exists(), f"Expected {saved}, dir contains: {list(save_dir.iterdir())}"
    assert saved.read_bytes() == SAMPLE_CONTENT


@pytest.mark.asyncio
async def test_download_with_explicit_filename(server, save_dir, session):
    """Download to an explicit filename (save_path has extension)."""
    url = f"{server}/download"
    target = save_dir / "custom_name.txt"

    result = await session.download(url, target, tab="default")
    assert "started" in result.lower()

    for _ in range(30):
        await asyncio.sleep(0.5)
        listing = session.download_list()
        if "complete" in listing:
            break
    else:
        pytest.fail(f"Download did not complete in time.\n{session.download_list()}")

    assert target.exists(), f"Expected {target}, dir contains: {list(save_dir.iterdir())}"
    assert target.read_bytes() == SAMPLE_CONTENT


@pytest.mark.asyncio
async def test_download_list_tracks_multiple(server, save_dir, session):
    """Multiple downloads appear in download_list."""
    url = f"{server}/download"

    r1 = await session.download(url, save_dir / "a", tab="default")
    r2 = await session.download(url, save_dir / "b", tab="default")
    assert "dl_1" in r1
    assert "dl_2" in r2

    for _ in range(30):
        await asyncio.sleep(0.5)
        listing = session.download_list()
        if listing.count("complete") >= 2:
            break
    else:
        pytest.fail(f"Downloads did not complete.\n{session.download_list()}")

    assert "dl_1" in listing
    assert "dl_2" in listing


@pytest.mark.asyncio
async def test_download_bad_selector_fails(server, save_dir, session):
    """Click-based download with bad selector reports failure."""
    await session.navigate(f"{server}/", tab="default")
    result = await session.download("#nonexistent", save_dir, tab="default")
    assert "started" in result.lower()

    for _ in range(80):
        await asyncio.sleep(0.5)
        listing = session.download_list()
        if "failed" in listing.lower():
            break
    else:
        pytest.fail(f"Expected failure but got:\n{session.download_list()}")
