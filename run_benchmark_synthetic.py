"""Synthetic benchmark command dispatcher.

This wrapper preserves the historical top-level entrypoint while the actual
implementation lives under ``src/scshapebench/synthetic``.

Available commands:
  - ``generate``: batch-generate a synthetic benchmark dataset
  - ``evaluate``: evaluate benchmark outputs
  - ``run-methods``: run the method benchmark on a dataset
  - ``run-swept-methods``: run the swept-dataset convenience wrapper

If no command is provided, ``run-swept-methods`` is used.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _dispatch(command: str, argv: list[str]) -> None:
    if command == "generate":
        from scshapebench.synthetic.batch_generate_benchmark import main
    elif command == "evaluate":
        from scshapebench.synthetic.evaluation import main
    elif command == "run-methods":
        from scshapebench.synthetic.method_benchmark.run_methods import main
    elif command == "run-swept-methods":
        from scshapebench.synthetic.method_benchmark.run_methods_on_swept_dataset import main
    else:
        raise SystemExit(
            f"Unknown synthetic command: {command!r}. "
            "Use one of: generate, evaluate, run-methods, run-swept-methods."
        )

    sys.argv = [sys.argv[0], *argv]
    main()


def main() -> None:
    args = sys.argv[1:]
    commands = {"generate", "evaluate", "run-methods", "run-swept-methods"}
    if args and args[0] in commands:
        command = args[0]
        argv = args[1:]
    else:
        command = "run-swept-methods"
        argv = args
    _dispatch(command, argv)


if __name__ == "__main__":
    main()
