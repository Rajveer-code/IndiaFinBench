"""
rag/embeddings.py
-----------------
Thin wrapper around sentence-transformers for BGE-base-en-v1.5.

BGE asymmetric encoding (important):
  - Corpus documents: encoded WITHOUT any prefix.
  - Queries: encoded WITH the prefix "Represent this sentence for searching
    relevant passages: " as specified by BAAI for bge-base-en-v1.5.
  Skipping the query prefix causes a measurable recall drop (~3–5% on MTEB).

All output embeddings are L2-normalised so that inner product = cosine similarity.
This is enforced here — callers must not re-normalise.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    def __init__(
        self,
        model_name:  str = "BAAI/bge-base-en-v1.5",
        device:      str = "cpu",
        batch_size:  int = 64,
    ) -> None:
        self._model      = SentenceTransformer(model_name, device=device)
        self.batch_size  = batch_size
        self.dim: int    = self._model.get_embedding_dimension()

    def encode_corpus(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of corpus texts.
        Returns float32 array of shape (len(texts), dim), L2-normalised.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query with the BGE query prefix.
        Returns float32 array of shape (1, dim), L2-normalised.
        """
        embedding = self._model.encode(
            _QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32).reshape(1, -1)
