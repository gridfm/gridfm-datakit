"""Atomic chunk checkpoints for distributed static generation."""

from copy import deepcopy
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from gridfm_datakit.utils.random_seed import _DEFAULT_SEED_POLICY


_DATASETS = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
    "y_bus_data.parquet",
    "runtime_data.parquet",
)
_NonNegativeInt = Annotated[int, Field(ge=0)]


class _CheckpointState(BaseModel):
    """Versioned configuration, progress, and committed output manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    seed: Annotated[int, Field(ge=0, le=_DEFAULT_SEED_POLICY.max_seed)]
    config: dict[str, Any]
    next_chunk: _NonNegativeInt = 0
    saved_scenarios: _NonNegativeInt = 0
    files: dict[str, _NonNegativeInt] = Field(default_factory=dict)


def _atomic_write(path: Path, content: str) -> None:
    """Flush a replacement file before atomically publishing it.

    Args:
        path: Destination in an existing output directory.
        content: UTF-8 text to persist.
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".checkpoint-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _resume_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return the settings that must match for deterministic continuation.

    Args:
        config: Validated generation configuration.
        seed: Saved effective seed, used when the requested seed is null.

    Returns:
        Configuration excluding operational settings that may change on resume.
    """
    result = deepcopy(config)
    settings = result["settings"]
    if settings.get("seed") is None:
        settings["seed"] = seed
    for name in ("overwrite", "data_dir", "enable_solver_logs", "process_backend"):
        settings.pop(name, None)
    return result


class GenerationCheckpoint:
    """Persist completed chunks and recover an interrupted parquet append.

    Attributes:
        root: Output directory containing checkpoint.json and the datasets.
        state: Last successfully published checkpoint.
    """

    def __init__(self, root: Path, state: _CheckpointState) -> None:
        """Bind a validated checkpoint to its output directory.

        Args:
            root: Generation output directory.
            state: Validated saved progress and file manifest.
        """
        self.root = root
        self.state = state

    @classmethod
    def load(
        cls,
        root: Path,
        config: dict[str, Any],
    ) -> Optional["GenerationCheckpoint"]:
        """Check resume eligibility before existing logs or output are changed.

        Args:
            root: Generation output directory.
            config: Validated requested configuration.

        Returns:
            Existing checkpoint, or None for a fresh run or ordinary append.

        Raises:
            ValueError: If settings, checkpoint state, or committed files disagree.
        """
        settings = config["settings"]
        if settings["overwrite"]:
            return None
        path = root / "checkpoint.json"
        enabled = settings.get("enable_checkpointing", False)
        if not path.exists():
            if enabled and any(
                (root / name).exists() for name in (*_DATASETS, "n_scenarios.txt")
            ):
                raise ValueError(
                    "Cannot resume existing output without checkpoint.json; use a new output directory or overwrite: true",
                )
            return None
        if not enabled:
            raise ValueError(
                "Existing output has a checkpoint; keep enable_checkpointing: true to resume or use overwrite: true",
            )
        try:
            state = _CheckpointState.model_validate_json(
                path.read_text(encoding="utf-8"),
            )
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot read checkpoint {path}: {error}") from error
        if _resume_config(config, state.seed) != state.config:
            raise ValueError(
                "Resume configuration differs from checkpoint; keep the original scenario, chunk, worker, seed, and solver settings",
            )
        chunks = (
            config["load"]["scenarios"] + settings["large_chunk_size"] - 1
        ) // settings["large_chunk_size"]
        if (
            state.next_chunk > chunks
            or (state.next_chunk == 0 and (state.files or state.saved_scenarios))
            or (state.files and not state.saved_scenarios)
        ):
            raise ValueError("Checkpoint contains inconsistent chunk progress")
        for name, size in state.files.items():
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 3
                or relative.parts[0] not in _DATASETS
                or not relative.parts[1].startswith("scenario_partition=")
                or relative.suffix != ".parquet"
            ):
                raise ValueError(f"Invalid committed file path in checkpoint: {name}")
            file = root / name
            if not file.is_file() or file.stat().st_size != size:
                raise ValueError(
                    f"Committed checkpoint file is missing or changed: {name}",
                )
        if state.saved_scenarios and any(
            not any(PurePosixPath(name).parts[0] == dataset for name in state.files)
            for dataset in _DATASETS
        ):
            raise ValueError("Checkpoint is missing a committed output dataset")
        return cls(root, state)

    @classmethod
    def create(
        cls,
        root: Path,
        config: dict[str, Any],
        seed: int,
    ) -> "GenerationCheckpoint":
        """Persist a fresh seed before any scenarios are generated.

        Args:
            root: Empty output directory prepared by generation setup.
            config: Validated generation configuration.
            seed: Effective random seed for this run.

        Returns:
            Newly published checkpoint with no completed chunks.
        """
        checkpoint = cls(
            root,
            _CheckpointState(seed=seed, config=_resume_config(config, seed)),
        )
        checkpoint._publish(checkpoint.state)
        return checkpoint

    def _files(self) -> dict[str, int]:
        """Return sizes of parquet files belonging to the five output datasets."""
        return {
            path.relative_to(self.root).as_posix(): path.stat().st_size
            for dataset in _DATASETS
            for path in (self.root / dataset).rglob("*.parquet")
        }

    def _publish(self, state: _CheckpointState) -> None:
        """Persist state before replacing the last committed in-memory state.

        Args:
            state: New progress and file manifest to publish atomically.
        """
        _atomic_write(self.root / "checkpoint.json", state.model_dump_json(indent=2))
        self.state = state

    def recover(self) -> None:
        """Remove uncommitted output files and restore the committed sample count."""
        for name in self._files().keys() - self.state.files.keys():
            (self.root / name).unlink()
        _atomic_write(self.root / "n_scenarios.txt", str(self.state.saved_scenarios))

    def commit(self, next_chunk: int) -> None:
        """Advance only after all dataset writes for a large chunk succeeded.

        Args:
            next_chunk: Index of the first large chunk not yet saved.
        """
        count = self.root / "n_scenarios.txt"
        saved = int(count.read_text(encoding="utf-8")) if count.exists() else 0
        self._publish(
            _CheckpointState(
                seed=self.state.seed,
                config=self.state.config,
                next_chunk=next_chunk,
                saved_scenarios=saved,
                files=self._files(),
            ),
        )
