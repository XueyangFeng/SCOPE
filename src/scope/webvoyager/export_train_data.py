from __future__ import annotations

import sys
from pathlib import Path

from scope.common.export_train_data import main as export_main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(Path(__file__).with_name("configs") / "export_train_data.yaml")])
    raise SystemExit(export_main())
