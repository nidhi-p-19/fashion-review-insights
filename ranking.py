# ranking.py
from typing import List, Dict
import numpy as np

# Lazy import so app can still run when ranking is OFF
def _lazy_imports():
    from sentence_transformers import SentenceTransformer, util
    return SentenceTransformer, util

_EMB_MODEL = None

def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        SentenceTransformer, _ = _lazy_imports()
        _EMB_MODEL = SentenceTransformer(model_name)
    return _EMB_MODEL

# Attribute queries help “relevance to topic”
ATTR_QUERIES = {
    "silhouette": "silhouette shape cut outline A-line boxy fitted flowy straight tapered",
    "proportion_or_fit": "fit proportion sizing true to size runs small runs large waist hip rise length",
    "detail": "detail stitching buttons zipper embroidery pockets hem seams",
    "color": "color dye shade fade wash brightness saturation",
    "print_or_pattern": "print pattern stripes polka dot floral check plaid graphic",
    "fabric": "fabric material cotton denim silk wool polyester viscose soft itchy thick thin stretch",
}

def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.ones_like(x) * 0.5
    return (x - lo) / (hi - lo)

def rank_attribute_snippets(
    sentences_block: List[Dict],
    attribute: str,
    top_k: int = 3,
    method: str = "weighted",   # "weighted" or "rrf"
    w_central: float = 0.6,
    w_relevance: float = 0.4
) -> List[Dict]:
    """
    sentences_block item format (from LLM):
      { "sentence": "...",
        "pairs": [ {"attribute":"fabric","snippet":"...","score":1}, ... ] }

    Returns Top-k items (dicts with sentence/snippet/sentiment/final_score).
    """
    # collect pairs for this attribute
    items = []
    for s in sentences_block or []:
        sent = (s.get("sentence") or "").strip()
        for p in s.get("pairs") or []:
            if p.get("attribute") == attribute:
                items.append({
                    "sentence": sent,
                    "snippet": (p.get("snippet") or sent).strip(),
                    "sentiment": int(p.get("score", 0))
                })
    if not items:
        return []

    # embeddings
    SentenceTransformer, util = _lazy_imports()
    embedder = _get_embedder()
    texts = [it["sentence"] for it in items]
    embs = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    # centrality (mean cosine sim to others)
    sim = util.cos_sim(embs, embs).cpu().numpy()
    n = sim.shape[0]
    centrality = (sim.sum(axis=1) - 1.0) / (n - 1) if n > 1 else np.ones(n)

    # relevance (cosine to attribute query)
    qtxt = ATTR_QUERIES.get(attribute, attribute)
    qemb = embedder.encode([qtxt], convert_to_tensor=True, normalize_embeddings=True)
    relevance = util.cos_sim(embs, qemb).cpu().numpy().ravel()

    if method == "weighted":
        c = _normalize(centrality)
        r = _normalize(relevance)
        final = w_central * c + w_relevance * r
    else:
        # RRF (if you later add another signal like KeyBERT)
        def to_ranks(x):
            order = np.argsort(-x)          # high score => low rank idx
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(x) + 1)
            return ranks.astype(float)

        k = 60.0
        parts = [to_ranks(centrality), to_ranks(relevance)]
        final = np.array([sum(1.0 / (k + p[i]) for p in parts) for i in range(n)], dtype=float)

    for i, it in enumerate(items):
        it["final_score"] = float(final[i])
    items.sort(key=lambda d: d["final_score"], reverse=True)
    return items[:top_k]
