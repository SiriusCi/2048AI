#!/usr/bin/env python3
"""Compatibility entrypoint for the Python-backed 2048 server."""

from backend.app import main, parse_args, serve
from backend.game import Game2048
from backend.http_handler import GameRequestHandler
from backend.service import GameService

__all__ = [
    "Game2048",
    "GameRequestHandler",
    "GameService",
    "main",
    "parse_args",
    "serve",
]


if __name__ == "__main__":
    raise SystemExit(main())
