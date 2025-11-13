from __future__ import annotations

import os
import sys
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from ui.cli import run_cli


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        # Provide minimal fallback when pyyaml is not available
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    log_dir = ((cfg.get("app") or {}).get("log_dir")) or "data/logs"
    vec_dir = ((cfg.get("rag") or {}).get("persist_dir")) or "data/vector_store"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vec_dir, exist_ok=True)


def main() -> None:
    cfg = load_config("config/settings.yaml")
    ensure_dirs(cfg)
    mode = ((cfg.get("app") or {}).get("mode")) or "cli"
    if mode == "cli":
        run_cli(cfg)
    else:
        print("当前示例仅实现 CLI 模式。可在 config/settings.yaml 中将 app.mode 设为 cli。")  # User-facing message, keep Chinese


if __name__ == "__main__":
    main()
