"""CSV-based training logger."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional


class CSVLogger:
    """Appends one row per call to a CSV file with auto-detected columns.

    The file is created on the first :meth:`log` call; the header row is
    written automatically from the keys of the first dict passed.

    Args:
        path: Destination file path.  Parent directories are created if needed.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", buffering=1)
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list[str]] = None

    def log(self, row: dict[str, Any]) -> None:
        """Write one row to the CSV.

        On the first call the header is written automatically.

        Args:
            row: Dict mapping column name → value.
        """
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"CSVLogger(path={self._path})"
