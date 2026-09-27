"""
Line-level semantic similarity between a source poem and a generated poem
(GRPO content reward, src/grpo_rewards.py; baseline in
src/build_reward_calibration.py).

BERTScore-style over lines rather than one whole-poem embedding: the
multilingual mpnet model truncates at 128 tokens (many poems are longer), and
a poem that only covers its first stanza's content should not score full
recall. Embedding model is frozen; source-line embeddings are cached by text
(each source recurs G times per GRPO step and across steps), generated lines
are not (they rarely repeat and would grow the cache without bound).
"""

import numpy as np

DEFAULT_CONTENT_MODEL = 'paraphrase-multilingual-mpnet-base-v2'


class LineEmbedder:
    def __init__(self, model_name: str = DEFAULT_CONTENT_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._cache: dict = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
        return self._model

    def _encode(self, lines: list) -> np.ndarray:
        return np.asarray(self._load().encode(
            lines, normalize_embeddings=True, batch_size=64, show_progress_bar=False),
            dtype=np.float32)

    def embed_lines(self, lines: list, cache: bool) -> np.ndarray:
        if not lines:
            dim = self._load().get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        if not cache:
            return self._encode(lines)
        missing = [l for l in dict.fromkeys(lines) if l not in self._cache]
        if missing:
            for line, vec in zip(missing, self._encode(missing)):
                self._cache[line] = vec
        return np.stack([self._cache[l] for l in lines])


def line_f1(src_emb: np.ndarray, gen_emb: np.ndarray) -> tuple:
    """(precision, recall, f1): recall = mean over source lines of their best
    match among generated lines; precision the reverse."""
    if len(src_emb) == 0 or len(gen_emb) == 0:
        return 0.0, 0.0, 0.0
    sim = src_emb @ gen_emb.T
    recall = float(sim.max(axis=1).mean())
    precision = float(sim.max(axis=0).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
