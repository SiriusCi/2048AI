"""Application bootstrap for the Python-backed 2048 server."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

from .config import load_runtime_config
from .http_handler import GameRequestHandler
from .rl import ReinforceCnnConfig
from .service import GameService


def serve(
    host: str,
    port: int,
    root_dir: str | Path | None = None,
    service: GameService | None = None,
) -> None:
    kwargs: dict[str, object] = {}
    if root_dir is not None:
        kwargs["root_dir"] = Path(root_dir).resolve()
    if service is not None:
        kwargs["service"] = service
    if kwargs:
        GameRequestHandler.configure(**kwargs)

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
    parser.add_argument("--config", default="config.yaml", help="YAML config path (default: config.yaml)")
    parser.add_argument("--host", default=None, help="Host override (default: config.server.host)")
    parser.add_argument("--port", default=None, type=int, help="Port override (default: config.server.port)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = load_runtime_config(args.config)
    host = args.host if args.host is not None else str(runtime["server"]["host"])
    port = int(args.port) if args.port is not None else int(runtime["server"]["port"])

    training_defaults = dict(runtime["trainingDefaults"])
    rl_raw = dict(runtime["rl"])
    rl_config = ReinforceCnnConfig(
        max_exponent=int(rl_raw["maxExponent"]),
        gamma=float(rl_raw["gamma"]),
        learning_rate=float(rl_raw["learningRate"]),
        entropy_coef=float(rl_raw["entropyCoef"]),
        value_coef=float(rl_raw["valueCoef"]),
        max_grad_norm=float(rl_raw["maxGradNorm"]),
        invalid_action_penalty=float(rl_raw["invalidActionPenalty"]),
        merge_value_bonus_scale=float(rl_raw["mergeValueBonusScale"]),
    )
    service = GameService(
        training_defaults=training_defaults,
        rl_config=rl_config,
        post_ack_delay_sec=float(runtime["sync"]["postAckDelaySec"]),
        default_tensorboard_log_dir=training_defaults.get("tensorboardLogDir"),
        default_checkpoint_dir=training_defaults.get("checkpointDir"),
    )
    print(f"Loaded config: {runtime['configPath']}")
    serve(host, port, service=service)
    return 0
