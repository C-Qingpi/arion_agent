#!/usr/bin/env python3
"""Verify semantic search cold-start fixes: per-file pipeline, warmup, agent lifecycle."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

pytest = __import__("pytest")
pytest.importorskip("fastembed")

# Optional: MarianMT for CJK path
try:
    import transformers  # noqa: F401
    has_mt = True
except ImportError:
    has_mt = False


def _fresh_ws(base: Path, *, with_cjk: bool = False) -> Path:
    ws = base / ("ws_cjk" if with_cjk else "ws_en")
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    (ws / "aaa_english.md").write_text(
        "The internal codename for the lunar relay project is Nebula Relay.\n",
        encoding="utf-8",
    )
    (ws / "bbb_config.md").write_text(
        "Deploy uses deploy.config with mode dev or prod.\n",
        encoding="utf-8",
    )
    if with_cjk:
        (ws / "ccc_chinese.md").write_text(
            "这是一个中文测试文档，用于验证翻译模型加载不会阻塞整个批次。\n",
            encoding="utf-8",
        )
    return ws


def test_per_file_first_manifest_fast(tmp_path: Path) -> None:
    from arion_agent.semantic_search.service import SearchService

    ws = _fresh_ws(tmp_path)
    svc = SearchService(ws)
    t0 = time.perf_counter()
    svc.start()

    first_at = None
    while time.perf_counter() - t0 < 30:
        st = svc.status()
        if st.indexed_files > 0:
            first_at = time.perf_counter() - t0
            break
        time.sleep(0.1)

    st = svc.status()
    svc.stop()
    assert first_at is not None, f"no file indexed in 30s: {st}"
    assert first_at < 5.0, f"first file too slow: {first_at:.2f}s (status={st})"
    print(f"PASS per_file_first_manifest: first file at {first_at:.2f}s, final {st.indexed_files}/{st.total_files}")


def test_agent_lifecycle_keeps_indexer(tmp_path: Path) -> None:
    from arion_agent.environments.search.middleware import SearchEnvironment

    ws = _fresh_ws(tmp_path)
    env = SearchEnvironment(ws, system_prompt=False)

    env.before_agent({})
    time.sleep(0.5)
    st1 = env.service.status()
    assert st1.thread_alive

    env.after_agent({})
    st2 = env.service.status()
    assert st2.thread_alive, "indexer must survive after_agent"

    env.before_agent({})
    time.sleep(1.0)
    st3 = env.service.status()
    assert st3.indexed_files >= st1.indexed_files
    env.service.stop()
    print(f"PASS agent_lifecycle: {st1.indexed_files} -> {st3.indexed_files} files")


def test_singleton_service_per_workspace(tmp_path: Path) -> None:
    from arion_agent.environments.search.middleware import SearchEnvironment

    ws = _fresh_ws(tmp_path)
    a = SearchEnvironment(ws, system_prompt=False)
    b = SearchEnvironment(ws, system_prompt=False)
    assert a.service is b.service
    print("PASS singleton_service")


def test_warmup_starts_embedder(tmp_path: Path) -> None:
    from arion_agent.semantic_search.embedder import embedder_loaded
    from arion_agent.semantic_search.service import SearchService

    ws = _fresh_ws(tmp_path)
    svc = SearchService(ws)
    svc.start()
    deadline = time.time() + 60
    while time.time() < deadline:
        if embedder_loaded():
            break
        time.sleep(0.2)
    svc.stop()
    assert embedder_loaded(), "embedder not loaded after warmup"
    print("PASS warmup_embedder")


def test_cjk_does_not_block_english_files(tmp_path: Path) -> None:
    if not has_mt:
        print("SKIP cjk_batch (transformers not installed)")
        return

    from arion_agent.semantic_search.service import SearchService

    ws = _fresh_ws(tmp_path, with_cjk=True)
    svc = SearchService(ws)
    t0 = time.perf_counter()
    svc.start()

    english_indexed_at = None
    while time.perf_counter() - t0 < 120:
        st = svc.status()
        manifest = svc.store.load_manifest()
        if "aaa_english.md" in manifest and english_indexed_at is None:
            english_indexed_at = time.perf_counter() - t0
        if st.initial_sync_done:
            break
        time.sleep(0.2)

    st = svc.status()
    manifest = svc.store.load_manifest()
    svc.stop()
    assert "aaa_english.md" in manifest, f"english file not indexed: {manifest.keys()}"
    assert english_indexed_at is not None
    assert english_indexed_at < 15.0, f"english file blocked too long: {english_indexed_at:.2f}s"
    print(
        f"PASS cjk_batch: english at {english_indexed_at:.2f}s, "
        f"final {st.indexed_files}/{st.total_files}"
    )


def test_agent_runner_warmup_hook() -> None:
    deploy = Path(__file__).resolve().parents[2] / "cross_platform_minimal_deploy"
    runner = deploy / "agent" / "agent_runner.py"
    text = runner.read_text(encoding="utf-8")
    assert "_warm_dev_search_indexers" in text
    assert "_warm_dev_search_indexers()" in text
    print("PASS agent_runner_warmup_hook")


def main() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="search_verify_"))
    tests = [
        test_singleton_service_per_workspace,
        test_warmup_starts_embedder,
        test_per_file_first_manifest_fast,
        test_agent_lifecycle_keeps_indexer,
        test_cjk_does_not_block_english_files,
        test_agent_runner_warmup_hook,
    ]
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            if fn is test_agent_runner_warmup_hook:
                fn()
            else:
                fn(tmp)
        except Exception as exc:
            print(f"FAIL {name}: {exc}")
            failed += 1
    shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        print(f"\n{failed} verification(s) failed")
        return 1
    print("\nAll verifications passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
