#!/usr/bin/env python3
"""End-to-end: warmup + immediate semantic_search on fresh multi-file workspace."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

pytest = __import__("pytest")
pytest.importorskip("fastembed")

DEPLOY = Path(__file__).resolve().parents[2] / "cross_platform_minimal_deploy"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="e2e_search_"))
    ws = tmp / "workspace"
    ws.mkdir()
    for i in range(12):
        (ws / f"doc_{i:02d}.md").write_text(
            f"Document {i} describes subsystem alpha-{i} and config mode dev.\n",
            encoding="utf-8",
        )
    (ws / "zzz_chinese.md").write_text(
        "这是中文文档，用于测试翻译不会阻塞英文文件的索引进度。\n",
        encoding="utf-8",
    )

    from arion_agent.environments.search.middleware import SearchEnvironment

    print("=== E2E: runner-style warmup + agent turn ===")
    t0 = time.perf_counter()

    # Same as _warm_search_indexers() + before_agent
    env = SearchEnvironment(ws, system_prompt=False)
    env.before_agent({})
    tool = env.tools[0]

    out0 = tool.invoke({"query": "subsystem alpha config"})
    t_immediate = time.perf_counter() - t0
    st0 = env.service.status()
    print(f"t={t_immediate:.2f}s immediate search: {out0[:140]!r}")
    print(f"  status: {st0.indexed_files}/{st0.total_files} alive={st0.thread_alive}")

    first_manifest = None
    while time.perf_counter() - t0 < 30:
        st = env.service.status()
        if st.indexed_files > 0 and first_manifest is None:
            first_manifest = time.perf_counter() - t0
        if st.indexed_files >= 5:
            break
        time.sleep(0.2)

    out1 = tool.invoke({"query": "subsystem alpha config"})
    t_later = time.perf_counter() - t0
    st1 = env.service.status()
    print(f"t={t_later:.2f}s search after progress: {out1[:140]!r}")
    print(f"  status: {st1.indexed_files}/{st1.total_files} chunks={st1.chunk_count}")

    env.after_agent({})
    st_after = env.service.status()
    print(f"after_agent alive={st_after.thread_alive} indexed={st_after.indexed_files}")

    env.before_agent({})
    time.sleep(0.5)
    st_turn2 = env.service.status()
    print(f"turn2 alive={st_turn2.thread_alive} indexed={st_turn2.indexed_files}")

    env.service.stop()
    shutil.rmtree(tmp, ignore_errors=True)

    assert st_after.thread_alive, "indexer died on after_agent"
    assert first_manifest is not None, "no file indexed within 30s"
    assert first_manifest < 8.0, f"first manifest too slow: {first_manifest:.2f}s"
    assert st1.indexed_files >= 5, f"expected >=5 indexed, got {st1.indexed_files}"
    assert st_turn2.indexed_files >= st1.indexed_files, "indexing regressed after turn2"
    assert "hits" in out1 or "doc_" in out1, f"expected hits: {out1[:200]}"
    print(f"\nPASS E2E first_file={first_manifest:.2f}s hits_at={t_later:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL E2E: {exc}")
        raise
