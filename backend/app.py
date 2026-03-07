"""Application bootstrap for the Python-backed 2048 server."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

from .http_handler import GameRequestHandler


def serve(host: str, port: int, root_dir: str | Path | None = None) -> None:
    if root_dir is not None:
        GameRequestHandler.configure(root_dir=Path(root_dir).resolve())

    server = ThreadingHTTPServer((host, port), GameRequestHandler)
    print(f"Serving 2048 backend on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Python-backed 2048 server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", default=8080, type=int, help="Port to bind (default: 8080)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    serve(args.host, args.port)
    return 0
