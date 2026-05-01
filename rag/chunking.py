"""
rag/chunking.py
---------------
Recursive character-level text splitter that respects paragraph, sentence,
and word boundaries — in that priority order.

Algorithm (same invariant as LangChain's RecursiveCharacterTextSplitter):
  1. Try each separator in SEPARATORS, highest-priority first.
  2. On the first separator found in the text, split and merge fragments
     back into chunks ≤ target_chunk_chars, carrying overlap_chars of
     context from the tail of each chunk into the next.
  3. Any merged chunk still over target_chunk_chars is recursively split
     with the remaining lower-priority separators.
  4. Chunks below min_chunk_chars (degenerate headers/footers) are discarded.

Separators prioritised for Indian regulatory text:
  \\n\\n > \\n > ". " > "; " > ", " > " " > "" (character-level fallback)
"""

from rag.models import ChunkRecord, Document


_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class RecursiveCharacterSplitter:
    def __init__(
        self,
        target_chunk_chars: int = 1600,
        overlap_chars:      int = 200,
        min_chunk_chars:    int = 100,
        separators:         list[str] | None = None,
    ) -> None:
        self.target   = target_chunk_chars
        self.overlap  = overlap_chars
        self.min_size = min_chunk_chars
        self.seps     = separators if separators is not None else _SEPARATORS

    # ── Public API ────────────────────────────────────────────────────────────

    def split_document(self, doc: Document) -> list[ChunkRecord]:
        fragments = self._split_recursive(doc.raw_text, self.seps)
        records: list[ChunkRecord] = []
        search_from = 0
        for idx, text in enumerate(fragments):
            # Best-effort character offset tracking.
            # Use the first 60 chars as a stable anchor since overlap means
            # the same text may appear twice near the split boundary.
            anchor = text[:60]
            pos = doc.raw_text.find(anchor, search_from)
            char_start = pos if pos != -1 else search_from
            char_end   = char_start + len(text)
            records.append(ChunkRecord(
                chunk_id   = f"{doc.doc_id}__{idx:04d}",
                doc_id     = doc.doc_id,
                title      = doc.title,
                source     = doc.source,
                text       = text,
                chunk_idx  = idx,
                char_start = char_start,
                char_end   = char_end,
            ))
            # Advance past this chunk, minus the overlap window
            search_from = max(0, char_end - self.overlap)
        return records

    # ── Core splitting logic ──────────────────────────────────────────────────

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.target:
            return [text] if len(text) >= self.min_size else []

        if not separators:
            # Character-level hard fallback: slice at target with overlap stride
            result: list[str] = []
            stride = max(1, self.target - self.overlap)
            for i in range(0, len(text), stride):
                chunk = text[i : i + self.target]
                if len(chunk) >= self.min_size:
                    result.append(chunk)
            return result

        sep, *remaining = separators

        if sep not in text:
            return self._split_recursive(text, remaining)

        # Split on this separator and merge into target-sized chunks
        frags   = text.split(sep)
        merged  = self._merge_with_overlap(frags, sep)

        # Recursively split any chunk still above target
        final: list[str] = []
        for chunk in merged:
            if len(chunk) > self.target and remaining:
                final.extend(self._split_recursive(chunk, remaining))
            else:
                final.append(chunk)
        return final

    def _merge_with_overlap(self, frags: list[str], sep: str) -> list[str]:
        """
        Merge a list of text fragments into chunks ≤ target_chars.
        After emitting a chunk, carry its last overlap_chars into the next
        chunk to preserve cross-boundary context.
        """
        chunks:      list[str]  = []
        current:     list[str]  = []   # fragments in the current chunk
        current_len: int        = 0

        for frag in frags:
            sep_cost   = len(sep) if current else 0
            addition   = sep_cost + len(frag)

            if current_len + addition > self.target and current:
                # Emit current chunk
                chunk_text = sep.join(current)
                if len(chunk_text) >= self.min_size:
                    chunks.append(chunk_text)

                # Carry overlap: walk backwards through current fragments
                overlap_frags: list[str] = []
                overlap_len:   int       = 0
                for f in reversed(current):
                    cost = (len(sep) if overlap_frags else 0) + len(f)
                    if overlap_len + cost > self.overlap:
                        break
                    overlap_frags.insert(0, f)
                    overlap_len += cost

                current     = overlap_frags + [frag]
                current_len = sum(
                    len(f) + (len(sep) if i > 0 else 0)
                    for i, f in enumerate(current)
                )
            else:
                current.append(frag)
                current_len += addition

        # Flush the last chunk
        if current:
            chunk_text = sep.join(current)
            if len(chunk_text) >= self.min_size:
                chunks.append(chunk_text)

        return chunks
