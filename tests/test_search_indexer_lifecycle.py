"""Smoke: indexer survives agent turns and reports progress."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("fastembed")


@pytest.fixture()
def search_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "alpha.md").write_text(
        "The internal codename for the lunar relay project is Nebula Relay.\n",
        encoding="utf-8",
    )
    (ws / "beta.md").write_text(
        "Deployment uses deploy.config with mode dev or prod.\n",
        encoding="utf-8",
    )
    yield ws
    idx = ws / ".arion" / "index"
    if idx.exists():
        shutil.rmtree(idx, ignore_errors=True)


def test_indexer_survives_agent_turns(search_ws: Path) -> None:
    from arion_agent.environments.search.middleware import SearchEnvironment

    env = SearchEnvironment(search_ws, system_prompt=False)
    tool = env.tools[0]

    env.before_agent({})
    deadline = time.time() + 60
    while time.time() < deadline:
        st = env.service.status()
        if st.indexed_files > 0:
            break
        time.sleep(0.25)

    st = env.service.status()
    assert st.thread_alive
    assert st.indexed_files > 0, f"expected indexed files, got {st}"
    first_count = st.indexed_files

    out = tool.invoke({"query": "lunar relay Nebula"})
    assert "Nebula" in out or "alpha.md" in out or st.indexed_files < st.total_files

    env.after_agent({})
    st_after = env.service.status()
    assert st_after.thread_alive, "indexer must keep running after agent turn"

    env.before_agent({})
    st2 = env.service.status()
    assert st2.thread_alive
    assert st2.indexed_files >= first_count

    env.service.stop()
