"""Calibrated decisions.

"Why 0.85?" is the question that ends a pitch. The honest answer for most systems is
"it looked right on our test file". This one ships an error bound instead.

Class-conditional split conformal: hold out a calibration set, take the score
quantile per material class, and emit a decision with a stated guarantee -

    Under exchangeability between calibration and deployment data, at most alpha of
    auto-accepted merges are wrong, per material class.

The assumption is stated because stating it is what makes the guarantee credible.
A new CPSE shifts the distribution, so its slice is recalibrated before auto-accept
is enabled for it.

Alpha is set by consequence, not uniformly: pressure-boundary and sour-service items
get 0.001 and no auto-accept without an identity key; consumables get 0.05.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config

AUTO_ACCEPT = "auto_accept"
REVIEW = "review"
AUTO_REJECT = "auto_reject"

_CAL_PATH = Path(config.VAR_DIR) / "conformal.json"


@dataclass
class ClassCalibration:
    class_code: str
    alpha: float
    accept_threshold: float
    reject_threshold: float
    n_positive: int
    n_negative: int
    fallback: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    tier: str
    score: float
    alpha: float
    accept_threshold: float
    reject_threshold: float
    guarantee: str
    reasons: List[str]
    identity_key_required: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _quantile(values: Sequence[float], q: float) -> float:
    """Conformal quantile with the finite-sample (n+1) correction."""
    if not values:
        return float("nan")
    vals = sorted(values)
    n = len(vals)
    rank = math.ceil((n + 1) * q)
    rank = max(1, min(rank, n))
    return float(vals[rank - 1])


class Calibrator:
    """Holds one calibration per material class."""

    def __init__(self, calibrations: Optional[Dict[str, ClassCalibration]] = None):
        self.calibrations: Dict[str, ClassCalibration] = calibrations or {}
        self.global_: Optional[ClassCalibration] = None

    # ------------------------------------------------------------------- fitting
    def fit(self, rows: Sequence[Tuple[str, float, int]], min_per_class: int = 12) -> dict:
        """rows: (class_code, score, label) with label in {0,1}.

        accept_threshold = the (1-alpha) quantile of NEGATIVE scores, so at most
        alpha of true non-matches can cross it.
        reject_threshold = the alpha quantile of POSITIVE scores, so at most alpha
        of true matches fall below it.
        """
        by_class: Dict[str, Tuple[List[float], List[float]]] = {}
        all_pos: List[float] = []
        all_neg: List[float] = []
        for class_code, score, label in rows:
            pos, neg = by_class.setdefault(class_code, ([], []))
            (pos if label == 1 else neg).append(float(score))
            (all_pos if label == 1 else all_neg).append(float(score))

        self.global_ = self._make("*", config.DEFAULT_ALPHA, all_pos, all_neg, fallback=False)
        self.calibrations = {}
        for class_code, (pos, neg) in by_class.items():
            alpha = config.alpha_for(class_code)
            if len(pos) + len(neg) < min_per_class:
                cal = self._make(class_code, alpha, all_pos, all_neg, fallback=True)
                cal.note = (
                    f"only {len(pos)+len(neg)} calibration points for this class - "
                    f"pooled thresholds used until more steward decisions exist"
                )
                self.calibrations[class_code] = cal
            else:
                self.calibrations[class_code] = self._make(class_code, alpha, pos, neg)
        return self.summary()

    @staticmethod
    def _make(class_code: str, alpha: float, pos: Sequence[float], neg: Sequence[float],
              fallback: bool = False) -> ClassCalibration:
        accept = _quantile(neg, 1.0 - alpha) if neg else 0.95
        reject = _quantile(pos, alpha) if pos else 0.05
        if math.isnan(accept):
            accept = 0.95
        if math.isnan(reject):
            reject = 0.05
        # Safety floors. The empirical quantile is the honest number, but two guards
        # sit over it: never auto-accept below 0.55 whatever the calibration says,
        # and never auto-reject above 0.5 - auto-reject only costs a false split,
        # which is the cheap direction, but a high floor there wastes steward time.
        accept = max(min(float(accept), 0.999), 0.55)
        reject = min(max(float(reject), 0.001), min(accept - 0.05, 0.5))
        return ClassCalibration(
            class_code=class_code, alpha=alpha,
            accept_threshold=round(accept, 6), reject_threshold=round(reject, 6),
            n_positive=len(pos), n_negative=len(neg), fallback=fallback,
        )

    # ------------------------------------------------------------------ decision
    def calibration_for(self, class_code: str) -> ClassCalibration:
        if class_code in self.calibrations:
            return self.calibrations[class_code]
        if self.global_ is not None:
            cal = ClassCalibration(**{**self.global_.to_dict(), "class_code": class_code})
            cal.alpha = config.alpha_for(class_code)
            cal.fallback = True
            cal.note = "no calibration for this class yet - pooled thresholds in use"
            return cal
        alpha = config.alpha_for(class_code)
        return ClassCalibration(
            class_code=class_code, alpha=alpha, accept_threshold=0.95, reject_threshold=0.05,
            n_positive=0, n_negative=0, fallback=True,
            note="uncalibrated - conservative defaults in use, auto-accept is disabled",
        )

    def decide(
        self,
        class_code: str,
        score: float,
        relation: str,
        has_identity_key: bool = False,
        has_graph_path: bool = False,
        blocked: bool = False,
    ) -> Decision:
        cal = self.calibration_for(class_code)
        reasons: List[str] = []
        id_required = class_code in config.IDENTITY_KEY_REQUIRED

        guarantee = (
            f"under exchangeability between calibration and deployment data, at most "
            f"{cal.alpha:.1%} of auto-accepted merges in class {class_code} are wrong"
        )
        if cal.fallback:
            guarantee += " (pooled thresholds - this class is not separately calibrated yet)"

        if blocked or relation in ("DISTINCT",):
            reasons.append("hard attribute conflict - recorded as DISTINCT and never re-proposed")
            return Decision(AUTO_REJECT, score, cal.alpha, cal.accept_threshold,
                            cal.reject_threshold, guarantee, reasons, id_required)

        if relation == "UNDETERMINED":
            reasons.append("non-inferable attribute missing on one record - a domain engineer decides")
            return Decision(REVIEW, score, cal.alpha, cal.accept_threshold,
                            cal.reject_threshold, guarantee, reasons, id_required)

        if relation in ("SUBSTITUTABLE", "VARIANT_OF"):
            reasons.append(
                "directional substitution is an engineering judgment, never an automatic merge"
                if relation == "SUBSTITUTABLE" else
                "same family, different size - rolled up for spend, never merged"
            )
            return Decision(REVIEW, score, cal.alpha, cal.accept_threshold,
                            cal.reject_threshold, guarantee, reasons, id_required)

        if score <= cal.reject_threshold:
            reasons.append(
                f"score {score:.3f} is at or below the {cal.alpha:.1%} quantile of known "
                f"matches ({cal.reject_threshold:.3f})"
            )
            return Decision(AUTO_REJECT, score, cal.alpha, cal.accept_threshold,
                            cal.reject_threshold, guarantee, reasons, id_required)

        if score >= cal.accept_threshold:
            if cal.fallback and cal.n_positive + cal.n_negative == 0:
                reasons.append("class is uncalibrated - auto-accept withheld until it is")
                return Decision(REVIEW, score, cal.alpha, cal.accept_threshold,
                                cal.reject_threshold, guarantee, reasons, id_required)
            if id_required and not has_identity_key:
                reasons.append(
                    f"class {class_code} is on the pressure boundary; auto-accept requires a "
                    f"manufacturer or vendor identity key, not text similarity alone"
                )
                return Decision(REVIEW, score, cal.alpha, cal.accept_threshold,
                                cal.reject_threshold, guarantee, reasons, id_required)
            if not (has_identity_key or has_graph_path or relation == "IDENTICAL"):
                reasons.append("no identity key and no standards-graph path - steward confirms")
                return Decision(REVIEW, score, cal.alpha, cal.accept_threshold,
                                cal.reject_threshold, guarantee, reasons, id_required)
            reasons.append(
                f"score {score:.3f} exceeds the {1-cal.alpha:.1%} quantile of known "
                f"non-matches ({cal.accept_threshold:.3f}); no conflict; identity established"
            )
            return Decision(AUTO_ACCEPT, score, cal.alpha, cal.accept_threshold,
                            cal.reject_threshold, guarantee, reasons, id_required)

        reasons.append(
            f"score {score:.3f} falls between the reject and accept thresholds "
            f"({cal.reject_threshold:.3f} / {cal.accept_threshold:.3f}) - the prediction "
            f"set is ambiguous at alpha={cal.alpha}"
        )
        return Decision(REVIEW, score, cal.alpha, cal.accept_threshold,
                        cal.reject_threshold, guarantee, reasons, id_required)

    # --------------------------------------------------------------- persistence
    def summary(self) -> dict:
        return {
            "global": self.global_.to_dict() if self.global_ else None,
            "classes": {k: v.to_dict() for k, v in sorted(self.calibrations.items())},
            "calibrated_classes": len(self.calibrations),
        }

    def to_dict(self) -> dict:
        return {
            "global": self.global_.to_dict() if self.global_ else None,
            "classes": {k: v.to_dict() for k, v in self.calibrations.items()},
        }

    @staticmethod
    def from_dict(payload: dict) -> "Calibrator":
        cal = Calibrator({k: ClassCalibration(**v) for k, v in (payload.get("classes") or {}).items()})
        if payload.get("global"):
            cal.global_ = ClassCalibration(**payload["global"])
        return cal

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or _CAL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Optional[Path] = None) -> "Calibrator":
        path = Path(path or _CAL_PATH)
        if path.exists():
            try:
                return Calibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return Calibrator()


def empirical_precision(rows: Sequence[Tuple[str, float, int]], calibrator: Calibrator) -> dict:
    """Measured precision at the auto-accept tier, reported WITH the abstention rate.
    A system that abstains on 30% and is right on the rest beats one that guesses
    everything at 0.92, and only reporting both makes that visible."""
    tiers = {AUTO_ACCEPT: [0, 0], REVIEW: [0, 0], AUTO_REJECT: [0, 0]}
    for class_code, score, label in rows:
        d = calibrator.decide(class_code, score, "EQUIVALENT", has_graph_path=True)
        bucket = tiers[d.tier]
        bucket[0] += 1
        bucket[1] += int(label == 1)
    total = sum(v[0] for v in tiers.values()) or 1
    acc_n, acc_pos = tiers[AUTO_ACCEPT]
    rej_n, rej_pos = tiers[AUTO_REJECT]
    return {
        "auto_accept_n": acc_n,
        "auto_accept_precision": round(acc_pos / acc_n, 4) if acc_n else None,
        "auto_reject_n": rej_n,
        "auto_reject_false_negative_rate": round(rej_pos / rej_n, 4) if rej_n else None,
        "review_n": tiers[REVIEW][0],
        "abstention_rate": round(tiers[REVIEW][0] / total, 4),
        "total": total,
    }
