"""Loading config/brain.json, and turning its relative paths into real ones."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "brain.json"


class Section(dict):
    """A config section that also allows `section.key`, for readability."""

    def __getattr__(self, name):
        try:
            value = self[name]

        except KeyError:
            raise AttributeError(name) from None

        return Section(value) if isinstance(value, dict) else value


def load(path=None):
    """Read the config, dropping the `_comment`-style keys."""
    data = json.loads(Path(path or CONFIG_PATH).read_text())

    def clean(value):
        if isinstance(value, dict):
            return Section(
                {key: clean(item) for key, item in value.items() if not key.startswith("_")}
            )

        return value

    return clean(data)


def path(relative):
    """A path from the config, relative to the repository root."""
    candidate = Path(relative).expanduser()

    return candidate if candidate.is_absolute() else ROOT / candidate
