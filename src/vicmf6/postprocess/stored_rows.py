"""Keep large intermediate tables on disk while summaries make repeated passes."""

import pickle
from contextlib import ExitStack
from pathlib import Path


class StoredRows:
    """Append and replay rows within one postprocessing run.

    Files live in the run's private temporary directory and are never accepted
    as input. Pickle preserves numeric types and None without a second schema.
    Each iterator opens its own reader, so independent passes cannot interfere.
    """

    def __init__(self, path: Path, resources: ExitStack):
        self.path = path
        self._writer = resources.enter_context(path.open("wb"))
        self._count = 0

    def append(self, row: dict) -> None:
        pickle.dump(row, self._writer, protocol=pickle.HIGHEST_PROTOCOL)
        self._count += 1

    def extend(self, rows) -> None:
        for row in rows:
            self.append(row)

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        self._writer.flush()
        with self.path.open("rb") as reader:
            for _ in range(self._count):
                yield pickle.load(reader)
