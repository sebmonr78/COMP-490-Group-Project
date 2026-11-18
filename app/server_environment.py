"""Load the dependency locations written by setup_server.sh, when available."""

import json
import os
from pathlib import Path
import warnings


def configure(root=None, environment=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    environment = os.environ if environment is None else environment
    settings = root / ".server/environment.json"
    if not settings.is_file():
        return
    try:
        config = json.loads(settings.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("expected an object")
        conda = config.get("conda_executable")
        dataset = config.get("dataset_root")
        if conda is not None and not isinstance(conda, str):
            raise ValueError("invalid Conda path")
        if dataset is not None and not isinstance(dataset, str):
            raise ValueError("invalid dataset path")
    except (OSError, ValueError) as error:
        warnings.warn("Could not load server dependency settings: %s. Rerun setup_server.sh." % error)
        return
    if conda and Path(conda).is_file() and os.access(conda, os.X_OK):
        environment.setdefault("CONDA_EXE", conda)
        folder = str(Path(environment["CONDA_EXE"]).parent)
        folders = environment.get("PATH", "").split(os.pathsep)
        if folder not in folders:
            environment["PATH"] = os.pathsep.join([folder] + folders)
    if dataset:
        environment.setdefault("HABITAT_DATA_PATH", dataset)
    if config.get("wsl_rendering"):
        environment.setdefault("GALLIUM_DRIVER", "d3d12")
        environment.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "d3d12")
