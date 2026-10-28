"""Dataset-specific locations beneath the shared dataset storage root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_NAME = "AstraZeneca–DREAM"


@dataclass(frozen=True)
class DatasetPaths:
    """Keep raw inputs, derived artifacts and manifests isolated by dataset."""

    root: Path
    name: str = DEFAULT_DATASET_NAME

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name in {".", ".."}
            or any(character in self.name for character in ("/", "\\", "\0"))
        ):
            raise ValueError(
                "dataset_name must be a nonempty directory name without path separators"
            )
        object.__setattr__(self, "root", Path(self.root))

    @property
    def raw(self) -> Path:
        return self.root / "raw" / self.name

    @property
    def processed(self) -> Path:
        return self.root / "processed" / self.name

    @property
    def splits(self) -> Path:
        return self.root / "splits" / self.name

    @property
    def manifests(self) -> Path:
        return self.root / "manifests" / self.name

    def source_path(self, relative: str) -> Path:
        """Resolve current or pre-migration DREAM raw provenance without rewriting it."""
        path = Path(relative)
        if (
            self.name == DEFAULT_DATASET_NAME
            and len(path.parts) >= 2
            and path.parts[0] == "raw"
            and path.parts[1] in {"az_dream", "az_dream_molecular", "cell_model_passports"}
        ):
            return self.raw.joinpath(*path.parts[1:])
        return self.root / path
