from __future__ import annotations

import sys
from pathlib import Path

from scope.cli.run_sequence import main as run_main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "webshop"])
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(Path(__file__).with_name("configs") / "sequence.yaml")])
    raise SystemExit(run_main())
