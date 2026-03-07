"""Backend package for the Python-driven 2048 server."""

from .app import main, parse_args, serve
from .game import Game2048
from .http_handler import GameRequestHandler
from .service import GameService

__all__ = [
    "Game2048",
    "Headless2048Env",
    "GameRequestHandler",
    "GameService",
    "main",
    "parse_args",
    "serve",
]


def __getattr__(name: str):
    if name == "Headless2048Env":
        from .headless import Headless2048Env

        return Headless2048Env
    raise AttributeError(f"module 'backend' has no attribute {name!r}")
