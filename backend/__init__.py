"""Backend package for the Python-driven 2048 server."""

from .app import main, parse_args, serve
from .game import Game2048
from .http_handler import GameRequestHandler
from .service import GameService

__all__ = [
    "Game2048",
    "GameRequestHandler",
    "GameService",
    "main",
    "parse_args",
    "serve",
]
