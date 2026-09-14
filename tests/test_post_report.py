import csv
from pathlib import Path

from vicmf6.post.report import write_csv


def test_write_csv_preserves_schema_for_empty_optional_table(tmp_path: Path):
    path = tmp_path / "empty.csv"
    write_csv(path, [], fieldnames=["model", "node", "value"])
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [["model", "node", "value"]]
