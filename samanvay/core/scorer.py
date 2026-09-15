"""The reranker.

Two backends behind one interface:

  features (default)  Logistic regression over the structured pair features, trained
                      with gradient descent on the labels mined from the ERP. Small,
                      fast, fully explainable, and it genuinely learns - the weights
                      move as stewards decide. No external dependency.

  cross               sentence-transformers CrossEncoder on attribute-serialised
                      pairs (Ditto-style: [COL] thread [VAL] M12x1.75 ...), for teams
                      that have the weights cached.

Shipped with defensible hand-set priors so the system works before it has seen a
single label, and improves from there.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config
from . import classify, features

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

# Hand-set priors. Signs encode the cost asymmetry: a conflict is expensive, a
# missing attribute is merely uninformative.
PRIOR_WEIGHTS: Dict[str, float] = {
    "bias": -3.4,
    "identity_key": 4.2,
    "attr_exact_ratio": 3.6,
    "attr_kg_ratio": 2.4,
    "attr_conflict_count": -3.1,
    "attr_missing_ratio": -0.9,
    "mandatory_covered": 2.8,
    "blocking_all_equal": 2.2,
    "numeric_match": 1.4,
    "numeric_mismatch": -3.4,
    "cosine": 1.8,
    "token_jaccard": 0.9,
    "trigram": 0.6,
    "same_class": 1.2,
    "service_unknown": -2.6,
    "standard_conflict": -1.3,
}

_MODEL_PATH = Path(config.VAR_DIR) / "scorer.json"
_CROSS = None


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticScorer:
    def __init__(self, weights: Optional[Sequence[float]] = None):
        if weights is None:
            weights = [PRIOR_WEIGHTS.get(name, 0.0) for name in features.FEATURE_NAMES]
        self.weights: List[float] = [float(w) for w in weights]
        self.trained_on: int = 0
        self.epochs: int = 0
        self.history: List[dict] = []

    # ---------------------------------------------------------------- inference
    def score(self, feats: Sequence[float]) -> float:
        z = sum(w * f for w, f in zip(self.weights, feats))
        return _sigmoid(z)

    def score_many(self, rows: Sequence[Sequence[float]]) -> List[float]:
        if _np is not None and rows:
            X = _np.asarray(rows, dtype=float)
            w = _np.asarray(self.weights, dtype=float)
            z = X.dot(w)
            return list(1.0 / (1.0 + _np.exp(-_np.clip(z, -30, 30))))
        return [self.score(r) for r in rows]

    # ----------------------------------------------------------------- training
    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        epochs: int = 400,
        lr: float = 0.35,
        l2: float = 0.002,
        class_weight_positive: float = 1.0,
        keep_prior: float = 0.25,
    ) -> dict:
        """Gradient descent with L2 and a pull towards the priors, so a handful of
        labels can refine the model without destroying domain knowledge."""
        if not X:
            return {"trained_on": 0, "loss": None}
        prior = [PRIOR_WEIGHTS.get(name, 0.0) for name in features.FEATURE_NAMES]
        n, d = len(X), len(X[0])

        if _np is not None:
            Xa = _np.asarray(X, dtype=float)
            ya = _np.asarray(y, dtype=float)
            wa = _np.asarray(self.weights, dtype=float)
            pa = _np.asarray(prior, dtype=float)
            sw = _np.where(ya > 0.5, class_weight_positive, 1.0)
            loss = None
            for _ in range(epochs):
                z = _np.clip(Xa.dot(wa), -30, 30)
                p = 1.0 / (1.0 + _np.exp(-z))
                err = (p - ya) * sw
                grad = Xa.T.dot(err) / n + l2 * wa + keep_prior * (wa - pa) / n
                wa = wa - lr * grad
                eps = 1e-9
                loss = float(-_np.mean(sw * (ya * _np.log(p + eps) + (1 - ya) * _np.log(1 - p + eps))))
            self.weights = [float(v) for v in wa]
        else:  # pure-python fallback
            w = list(self.weights)
            loss = None
            for _ in range(epochs):
                grad = [0.0] * d
                total = 0.0
                for xi, yi in zip(X, y):
                    p = _sigmoid(sum(wj * xj for wj, xj in zip(w, xi)))
                    sw = class_weight_positive if yi > 0.5 else 1.0
                    err = (p - yi) * sw
                    for j in range(d):
                        grad[j] += err * xi[j]
                    eps = 1e-9
                    total += -sw * (yi * math.log(p + eps) + (1 - yi) * math.log(1 - p + eps))
                for j in range(d):
                    grad[j] = grad[j] / n + l2 * w[j] + keep_prior * (w[j] - prior[j]) / n
                    w[j] -= lr * grad[j]
                loss = total / n
            self.weights = w

        self.trained_on = n
        self.epochs += epochs
        metrics = self.evaluate(X, y)
        entry = {"trained_on": n, "loss": round(loss, 6) if loss is not None else None, **metrics}
        self.history.append(entry)
        return entry

    def evaluate(self, X: Sequence[Sequence[float]], y: Sequence[int], threshold: float = 0.5) -> dict:
        if not X:
            return {}
        scores = self.score_many(X)
        tp = sum(1 for s, t in zip(scores, y) if s >= threshold and t == 1)
        fp = sum(1 for s, t in zip(scores, y) if s >= threshold and t == 0)
        fn = sum(1 for s, t in zip(scores, y) if s < threshold and t == 1)
        tn = sum(1 for s, t in zip(scores, y) if s < threshold and t == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "feature_names": list(features.FEATURE_NAMES),
            "weights": self.weights,
            "trained_on": self.trained_on,
            "epochs": self.epochs,
            "history": self.history[-20:],
        }

    @staticmethod
    def from_dict(payload: dict) -> "LogisticScorer":
        names = payload.get("feature_names") or list(features.FEATURE_NAMES)
        by_name = dict(zip(names, payload.get("weights") or []))
        weights = [by_name.get(n, PRIOR_WEIGHTS.get(n, 0.0)) for n in features.FEATURE_NAMES]
        model = LogisticScorer(weights)
        model.trained_on = int(payload.get("trained_on") or 0)
        model.epochs = int(payload.get("epochs") or 0)
        model.history = list(payload.get("history") or [])
        return model

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or _MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Optional[Path] = None) -> "LogisticScorer":
        path = Path(path or _MODEL_PATH)
        if path.exists():
            try:
                return LogisticScorer.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return LogisticScorer()

    def weight_table(self) -> List[dict]:
        prior = [PRIOR_WEIGHTS.get(n, 0.0) for n in features.FEATURE_NAMES]
        return [
            {"feature": n, "weight": round(w, 4), "prior": round(p, 4), "moved": round(w - p, 4)}
            for n, w, p in zip(features.FEATURE_NAMES, self.weights, prior)
        ]


# ------------------------------------------------------------------ cross-encoder

def _serialise(record: dict) -> str:
    """Ditto-style structured serialisation - the model sees fields, not a sentence."""
    cls = record.get("class_code", "GENERIC")
    parts = [f"[COL] class [VAL] {cls}"]
    for spec in classify.attribute_specs(cls):
        value = (record.get("attributes") or {}).get(spec["key"])
        if value:
            parts.append(f"[COL] {spec['key']} [VAL] {value}")
    parts.append(f"[COL] text [VAL] {record.get('description','')}")
    return " ".join(parts)


def _cross_model():
    global _CROSS
    if _CROSS is None:
        from sentence_transformers import CrossEncoder  # noqa: WPS433
        _CROSS = CrossEncoder(config.RERANK_MODEL)
    return _CROSS


def backend() -> str:
    if config.RERANK_BACKEND == "cross":
        try:
            _cross_model()
            return "cross"
        except Exception:
            return "features"
    return "features"


def cross_scores(pairs: Sequence[Tuple[dict, dict]]) -> Optional[List[float]]:
    if backend() != "cross" or not pairs:
        return None
    try:
        model = _cross_model()
        raw = model.predict([(_serialise(a), _serialise(b)) for a, b in pairs])
        return [float(1.0 / (1.0 + math.exp(-float(r)))) if abs(float(r)) > 1.0 else float(r) for r in raw]
    except Exception:
        return None
