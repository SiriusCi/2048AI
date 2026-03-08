"""Runtime configuration loading from YAML."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency at runtime
    yaml = None  # type: ignore[assignment]


DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8080,
    },
    "sync": {
        "postAckDelaySec": 0.1,
    },
    "trainingDefaults": {
        "episodes": 100,
        "workers": 1,
        "seed": None,
        "maxSteps": None,
        "terminateOnWin": True,
        "tensorboardLogDir": "runs/2048",
        "tensorboardRunName": None,
        "checkpointEveryEpisodes": 0,
        "checkpointDir": "models/2048",
        "checkpointPrefix": "reinforce_cnn",
        "loadModelPath": None,
        "playOnly": False,
    },
    "rl": {
        "maxExponent": 15,
        "gamma": 0.99,
        "learningRate": 3e-4,
        "entropyCoef": 1e-3,
        "invalidActionPenalty": -1.0,
        "mergeValueBonusScale": 1.0,
    },
}


def default_runtime_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_RUNTIME_CONFIG)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _as_int(value: Any, *, field: str, min_value: int | None = None) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Config field '{field}' must be an integer.") from error
    if min_value is not None and converted < min_value:
        raise ValueError(f"Config field '{field}' must be >= {min_value}.")
    return converted


def _as_optional_int(value: Any, *, field: str, min_value: int | None = None) -> int | None:
    if value is None:
        return None
    return _as_int(value, field=field, min_value=min_value)


def _as_float(value: Any, *, field: str, min_value: float | None = None) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Config field '{field}' must be a number.") from error
    if min_value is not None and converted < min_value:
        raise ValueError(f"Config field '{field}' must be >= {min_value}.")
    return converted


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config field '{field}' must be true/false.")


def _as_str(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Config field '{field}' must be a string.")
    if not allow_empty and value.strip() == "":
        raise ValueError(f"Config field '{field}' must be non-empty.")
    return value


def _as_optional_str(value: Any, *, field: str, allow_empty: bool = False) -> str | None:
    if value is None:
        return None
    return _as_str(value, field=field, allow_empty=allow_empty)


def _resolve_config_path(path: str | Path) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return (Path.cwd() / path_obj).resolve()


def _load_raw_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml. Install with `pip install pyyaml`.")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML config root must be a mapping/object.")
    return loaded


def load_runtime_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = _resolve_config_path(path)
    raw = _load_raw_yaml(config_path)
    merged = _deep_merge(default_runtime_config(), raw)

    server = merged.get("server", {})
    sync = merged.get("sync", {})
    training_defaults = merged.get("trainingDefaults", {})
    rl = merged.get("rl", {})

    merge_value_bonus_scale_raw = rl.get("mergeValueBonusScale", rl.get("mergeEmptyReductionBonus", 1.0))

    normalized: dict[str, Any] = {
        "configPath": str(config_path),
        "server": {
            "host": _as_str(server.get("host"), field="server.host"),
            "port": _as_int(server.get("port"), field="server.port", min_value=1),
        },
        "sync": {
            "postAckDelaySec": _as_float(sync.get("postAckDelaySec"), field="sync.postAckDelaySec", min_value=0.0),
        },
        "trainingDefaults": {
            "episodes": _as_int(training_defaults.get("episodes"), field="trainingDefaults.episodes", min_value=1),
            "workers": _as_int(training_defaults.get("workers"), field="trainingDefaults.workers", min_value=0),
            "seed": _as_optional_int(training_defaults.get("seed"), field="trainingDefaults.seed"),
            "maxSteps": _as_optional_int(training_defaults.get("maxSteps"), field="trainingDefaults.maxSteps", min_value=1),
            "terminateOnWin": _as_bool(training_defaults.get("terminateOnWin"), field="trainingDefaults.terminateOnWin"),
            "tensorboardLogDir": _as_optional_str(
                training_defaults.get("tensorboardLogDir"),
                field="trainingDefaults.tensorboardLogDir",
                allow_empty=False,
            ),
            "tensorboardRunName": _as_optional_str(
                training_defaults.get("tensorboardRunName"),
                field="trainingDefaults.tensorboardRunName",
                allow_empty=False,
            ),
            "checkpointEveryEpisodes": _as_int(
                training_defaults.get("checkpointEveryEpisodes"),
                field="trainingDefaults.checkpointEveryEpisodes",
                min_value=0,
            ),
            "checkpointDir": _as_optional_str(
                training_defaults.get("checkpointDir"),
                field="trainingDefaults.checkpointDir",
                allow_empty=False,
            ),
            "checkpointPrefix": _as_str(
                training_defaults.get("checkpointPrefix"),
                field="trainingDefaults.checkpointPrefix",
                allow_empty=False,
            ),
            "loadModelPath": _as_optional_str(
                training_defaults.get("loadModelPath"),
                field="trainingDefaults.loadModelPath",
                allow_empty=False,
            ),
            "playOnly": _as_bool(training_defaults.get("playOnly"), field="trainingDefaults.playOnly"),
        },
        "rl": {
            "maxExponent": _as_int(rl.get("maxExponent"), field="rl.maxExponent", min_value=1),
            "gamma": _as_float(rl.get("gamma"), field="rl.gamma", min_value=0.0),
            "learningRate": _as_float(rl.get("learningRate"), field="rl.learningRate", min_value=0.0),
            "entropyCoef": _as_float(rl.get("entropyCoef"), field="rl.entropyCoef", min_value=0.0),
            "invalidActionPenalty": _as_float(rl.get("invalidActionPenalty"), field="rl.invalidActionPenalty"),
            "mergeValueBonusScale": _as_float(
                merge_value_bonus_scale_raw,
                field="rl.mergeValueBonusScale",
                min_value=0.0,
            ),
        },
    }
    return normalized
