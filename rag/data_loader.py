"""
rag/data_loader.py
------------------
Scan data/parsed/{rbi,sebi}/*.txt and return a flat list of Document objects.
Title extraction parses the filename convention used by the IndiaFinBench corpus:
    {SOURCE}_{ShortLabel}_{full_slug}_{index}.txt
e.g. RBI_Master Dir_master_direction_-_..._084.txt → title "RBI — Master Dir"
"""

import re
from pathlib import Path

from rag.models import Document


class DataLoader:
    SOURCES = ("rbi", "sebi")

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)

    def load(self) -> list[Document]:
        docs: list[Document] = []
        for source in self.SOURCES:
            source_dir = self.data_dir / source
            if not source_dir.exists():
                continue
            for fpath in sorted(source_dir.glob("*.txt")):
                text = fpath.read_text(encoding="utf-8", errors="replace")
                docs.append(Document(
                    doc_id=fpath.stem,
                    title=self._parse_title(fpath.stem, source),
                    source=source,
                    raw_text=text,
                    file_path=str(fpath.resolve()),
                ))
        return docs

    @staticmethod
    def _parse_title(stem: str, source: str) -> str:
        """
        Filename format: {SRC}_{ShortLabel}_{long_slug}_{index}
        Extract the short label (second underscore-delimited field).
        """
        parts = stem.split("_", 2)
        if len(parts) >= 2:
            label = parts[1].strip()
            # Collapse runs of spaces introduced by the naming convention
            label = re.sub(r"\s+", " ", label)
            return f"{source.upper()} — {label}"
        return stem
