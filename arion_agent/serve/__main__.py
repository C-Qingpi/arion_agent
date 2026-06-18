"""Standalone entry point for the arion_agent I/O + checkpoint service.

Usage:
    python -m arion_agent.serve --root /path/to/workspace --port 8911
    python -m arion_agent.serve --root /workspace --checkpoint-db /path/to/checkpoints.sqlite
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArionAgent remote I/O and checkpoint service"
    )
    parser.add_argument("--root", required=True, help="Workspace root directory")
    parser.add_argument("--checkpoint-db", default=None,
                        help="Path to checkpoint SQLite database")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8911, help="Port number")
    args = parser.parse_args()

    import uvicorn
    from fastapi import FastAPI

    from arion_agent.serve import create_service_router

    app = FastAPI(title="ArionAgent I/O Service")
    router = create_service_router(args.root, args.checkpoint_db)
    app.include_router(router)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
