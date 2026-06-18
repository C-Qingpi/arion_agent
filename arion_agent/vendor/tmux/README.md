# Bundled tmux prefixes (conda-forge layout: bin/tmux + lib/)

Populated by maintainers:

```bash
python scripts/vendor_tmux.py
```

Run once per target OS/CPU (darwin-arm64, linux-x86_64, …). Commit the resulting
``bin/`` and ``lib/`` trees. Runtime resolves ``arion_agent/vendor/tmux/<platform>/``
automatically; system tmux on PATH is a fallback only.
