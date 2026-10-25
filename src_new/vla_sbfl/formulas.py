"""SBFL formula registry.

Table 4 notation is mapped as follows::

    N_CF = ef   covered by failed episodes
    N_UF = nf   not covered by failed episodes
    N_CS = ep   covered by successful episodes
    N_US = np   not covered by successful episodes

Every evaluator returns finite scores plus an explicit per-voxel status. A
mathematical 0/0 is encoded with the formula's documented neutral value. A
true positive or negative infinity is replaced by a finite sentinel that
preserves its extreme rank within that fitted map.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    """The four binary-coverage contingency counts for one or more voxels."""

    ef: np.ndarray
    ep: np.ndarray
    nf: np.ndarray
    npass: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(value, dtype=np.float64) for value in self.values]
        shape = arrays[0].shape
        if any(value.shape != shape for value in arrays):
            raise ValueError("all spectrum arrays must have the same shape")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("spectrum counts must be finite")
        if any((value < 0).any() for value in arrays):
            raise ValueError("spectrum counts cannot be negative")
        object.__setattr__(self, "ef", arrays[0])
        object.__setattr__(self, "ep", arrays[1])
        object.__setattr__(self, "nf", arrays[2])
        object.__setattr__(self, "npass", arrays[3])

    @property
    def values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.ef, self.ep, self.nf, self.npass

    @property
    def covered(self) -> np.ndarray:
        return self.ef + self.ep

    @property
    def uncovered(self) -> np.ndarray:
        return self.nf + self.npass

    @property
    def failed(self) -> np.ndarray:
        return self.ef + self.nf

    @property
    def passed(self) -> np.ndarray:
        return self.ep + self.npass

    @property
    def total(self) -> np.ndarray:
        return self.covered + self.uncovered

    @property
    def determinant(self) -> np.ndarray:
        return self.ef * self.npass - self.ep * self.nf

    def validate_training_domain(self) -> None:
        if self.ef.size == 0:
            raise ValueError("cannot evaluate formulas without voxels")
        if (self.covered <= 0).any():
            raise ValueError("training-covered voxels must have positive support")
        if (self.total <= 0).any():
            raise ValueError("training data must contain at least one episode")


FormulaFunction = Callable[[Spectrum], np.ndarray]


@dataclass(frozen=True)
class FormulaSpec:
    formula_id: str
    name: str
    expression: str
    families: Tuple[str, ...]
    evaluator: FormulaFunction
    undefined_value: float = 0.0


@dataclass(frozen=True)
class FormulaEvaluation:
    scores: np.ndarray
    statuses: np.ndarray
    undefined_count: int
    positive_infinity_count: int
    negative_infinity_count: int


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.asarray(numerator, dtype=np.float64) / np.asarray(denominator, dtype=np.float64)


def _table4_specs() -> Sequence[FormulaSpec]:
    t = ("table4",)
    return (
        FormulaSpec("braun_banquet", "Braun-Banquet", "ef / max(ef+ep, ef+nf)", t,
                    lambda s: _divide(s.ef, np.maximum(s.covered, s.failed))),
        FormulaSpec("dennis", "Dennis", "(ef*np-ep*nf) / sqrt(n*(ef+ep)*(ef+nf))", t,
                    lambda s: _divide(s.determinant, np.sqrt(s.total * s.covered * s.failed))),
        FormulaSpec("mountford", "Mountford", "ef / (0.5*(ef*ep+ef*nf)+ep*nf)", t,
                    lambda s: _divide(s.ef, 0.5 * (s.ef*s.ep + s.ef*s.nf) + s.ep*s.nf)),
        FormulaSpec("fossum", "Fossum", "n*(ef-0.5)^2 / ((ef+ep)*(ef+nf))", t,
                    lambda s: _divide(s.total * (s.ef - 0.5)**2, s.covered * s.failed)),
        FormulaSpec("pearson", "Pearson", "n*(ef*np-ep*nf)^2 / (C*U*S*F)", t,
                    lambda s: _divide(s.total * s.determinant**2,
                                      s.covered*s.uncovered*s.passed*s.failed)),
        FormulaSpec("gower", "Gower", "(ef+np) / sqrt(F*C*U*S)", t,
                    lambda s: _divide(s.ef+s.npass,
                                      np.sqrt(s.failed*s.covered*s.uncovered*s.passed))),
        FormulaSpec("michael", "Michael", "4*(ef*np-ep*nf) / ((ef+np)^2+(ep+nf)^2)", t,
                    lambda s: _divide(4*s.determinant, (s.ef+s.npass)**2+(s.ep+s.nf)**2)),
        FormulaSpec("pierce", "Pierce", "(ef*nf+nf*ep) / (ef*nf+2*nf*np+ep*np)", t,
                    lambda s: _divide(s.ef*s.nf+s.nf*s.ep,
                                      s.ef*s.nf+2*s.nf*s.npass+s.ep*s.npass)),
        FormulaSpec("baroni_urbani_buser", "Baroni-Urbani & Buser",
                    "(sqrt(ef*np)+ef) / (sqrt(ef*np)+ef+ep+nf)", t,
                    lambda s: _divide(np.sqrt(s.ef*s.npass)+s.ef,
                                      np.sqrt(s.ef*s.npass)+s.ef+s.ep+s.nf)),
        FormulaSpec("tarwid", "Tarwid", "(n*ef-F*C) / (n*ef+F*C)", t,
                    lambda s: _divide(s.total*s.ef-s.failed*s.covered,
                                      s.total*s.ef+s.failed*s.covered)),
        FormulaSpec("ample", "Ample", "abs(ef/F - ep/S)", ("table4", "legacy14"),
                    lambda s: np.abs(_divide(s.ef, s.failed)-_divide(s.ep, s.passed))),
        FormulaSpec("phi", "Phi (Geometric Mean)",
                    "(ef*np-nf*ep) / sqrt(C*F*S*U)", t,
                    lambda s: _divide(s.determinant,
                                      np.sqrt(s.covered*s.failed*s.passed*s.uncovered))),
        FormulaSpec("arithmetic_mean", "Arithmetic Mean",
                    "2*(ef*np-nf*ep) / (C*U+F*S)", t,
                    lambda s: _divide(2*s.determinant,
                                      s.covered*s.uncovered+s.failed*s.passed)),
        FormulaSpec("cohen", "Cohen", "2*(ef*np-nf*ep) / (C*S+F*U)", t,
                    lambda s: _divide(2*s.determinant,
                                      s.covered*s.passed+s.failed*s.uncovered)),
        FormulaSpec("fleiss", "Fleiss",
                    "(4*(ef*np-nf*ep)-(nf-ep)^2) / ((2ef+nf+ep)+(2np+nf+ep))", t,
                    lambda s: _divide(4*s.determinant-(s.nf-s.ep)**2,
                                      (2*s.ef+s.nf+s.ep)+(2*s.npass+s.nf+s.ep))),
        FormulaSpec("zoltar", "Zoltar", "ef / (ef+nf+ep+10000*nf*ep/ef)",
                    ("table4", "legacy14"),
                    lambda s: _divide(s.ef, s.ef+s.nf+s.ep+_divide(10000*s.nf*s.ep, s.ef))),
        FormulaSpec("harmonic_mean", "Harmonic Mean",
                    "det*(C*U+F*S) / (C*U*F*S)", t,
                    lambda s: _divide(s.determinant*(s.covered*s.uncovered+s.failed*s.passed),
                                      s.covered*s.uncovered*s.failed*s.passed)),
        FormulaSpec("rogot2", "Rogot2", "1/4*(ef/C+ef/F+np/S+np/U)", t,
                    lambda s: 0.25*(_divide(s.ef,s.covered)+_divide(s.ef,s.failed)
                                    +_divide(s.npass,s.passed)+_divide(s.npass,s.uncovered))),
        FormulaSpec("simple_matching", "Simple Matching", "(ef+np)/n", t,
                    lambda s: _divide(s.ef+s.npass, s.total)),
        FormulaSpec("rogers_tanimoto", "Rogers & Tanimoto",
                    "(ef+np) / (ef+np+2*(nf+ep))", t,
                    lambda s: _divide(s.ef+s.npass, s.ef+s.npass+2*(s.nf+s.ep))),
        FormulaSpec("hamming", "Hamming", "ef+np", t, lambda s: s.ef+s.npass),
        FormulaSpec("hamann", "Hamann", "(ef+np-nf-ep)/n", t,
                    lambda s: _divide(s.ef+s.npass-s.nf-s.ep, s.total)),
        FormulaSpec("sokal", "Sokal", "2*(ef+np) / (2*(ef+np)+nf+ep)", t,
                    lambda s: _divide(2*(s.ef+s.npass), 2*(s.ef+s.npass)+s.nf+s.ep)),
        FormulaSpec("scott", "Scott",
                    "(4*(ef*np-nf*ep)-(nf-ep)^2) / ((2ef+nf+ep)*(2np+nf+ep))", t,
                    lambda s: _divide(4*s.determinant-(s.nf-s.ep)**2,
                                      (2*s.ef+s.nf+s.ep)*(2*s.npass+s.nf+s.ep))),
        FormulaSpec("rogot1", "Rogot1",
                    "1/2*(ef/(2ef+nf+ep)+np/(2np+nf+ep))", t,
                    lambda s: 0.5*(_divide(s.ef,2*s.ef+s.nf+s.ep)
                                   +_divide(s.npass,2*s.npass+s.nf+s.ep))),
        FormulaSpec("kulczynski", "Kulczynski", "ef/(nf+ep)", t,
                    lambda s: _divide(s.ef, s.nf+s.ep)),
        FormulaSpec("anderberg", "Anderberg", "ef/(ef+2*(nf+ep))", t,
                    lambda s: _divide(s.ef, s.ef+2*(s.nf+s.ep))),
        FormulaSpec("dice", "Dice", "2ef/(ef+nf+ep)", t,
                    lambda s: _divide(2*s.ef, s.ef+s.nf+s.ep)),
        FormulaSpec("goodman", "Goodman", "(2ef-nf-ep)/(2ef+nf+ep)", t,
                    lambda s: _divide(2*s.ef-s.nf-s.ep, 2*s.ef+s.nf+s.ep)),
        FormulaSpec("jaccard", "Jaccard", "ef/(ef+nf+ep)", ("table4", "legacy14"),
                    lambda s: _divide(s.ef, s.ef+s.nf+s.ep)),
        FormulaSpec("sorensen_dice", "Sorensen-Dice", "2ef/(2ef+nf+ep)", t,
                    lambda s: _divide(2*s.ef, 2*s.ef+s.nf+s.ep)),
    )


def _legacy_only_specs() -> Sequence[FormulaSpec]:
    family = ("legacy14",)

    def wong3(s: Spectrum) -> np.ndarray:
        h = np.where(s.ep <= 2, s.ep,
                     np.where(s.ep <= 10, 2+0.1*(s.ep-2), 2.8+0.001*(s.ep-10)))
        return s.ef-h

    return (
        FormulaSpec("tarantula", "Tarantula", "(ef/F)/((ef/F)+(ep/S))",
                    ("legacy14", "study_extension"),
                    lambda s: _divide(_divide(s.ef,s.failed),
                                      _divide(s.ef,s.failed)+_divide(s.ep,s.passed))),
        FormulaSpec("barinel", "Barinel", "ef/(ef+ep)", family,
                    lambda s: _divide(s.ef, s.covered)),
        FormulaSpec("kulczynski2", "Kulczynski2", "0.5*(ef/F+ef/(ef+ep))", family,
                    lambda s: 0.5*(_divide(s.ef,s.failed)+_divide(s.ef,s.covered))),
        FormulaSpec("wong2", "Wong2", "ef-ep", family, lambda s: s.ef-s.ep),
        FormulaSpec("ochiai2", "Ochiai2", "ef*np/sqrt(C*U*F*S)",
                    ("legacy14", "study_extension"),
                    lambda s: _divide(s.ef*s.npass,
                                      np.sqrt(s.covered*s.uncovered*s.failed*s.passed))),
        FormulaSpec("ochiai", "Ochiai", "ef/sqrt(F*C)",
                    ("legacy14", "study_extension"),
                    lambda s: _divide(s.ef, np.sqrt(s.failed*s.covered))),
        FormulaSpec("dstar2", "DStar2", "ef^2/(ep+nf)", family,
                    lambda s: _divide(s.ef**2, s.ep+s.nf)),
        FormulaSpec("wong3", "Wong3", "ef-h(ep)", family, wong3),
        FormulaSpec("gp13", "GP13", "ef*(1+1/(ef+2ep))", family,
                    lambda s: s.ef*(1+_divide(1, s.ef+2*s.ep))),
        FormulaSpec("op2", "OP2", "ef-ep/(S+1)", family,
                    lambda s: s.ef-_divide(s.ep,s.passed+1)),
        FormulaSpec("wong1", "Wong1", "ef", family, lambda s: s.ef.copy()),
    )


FORMULA_REGISTRY: Mapping[str, FormulaSpec] = {
    spec.formula_id: spec for spec in (*_table4_specs(), *_legacy_only_specs())
}

TABLE4_FORMULA_IDS: Tuple[str, ...] = tuple(
    spec.formula_id for spec in _table4_specs()
)
STUDY_FORMULA_IDS: Tuple[str, ...] = TABLE4_FORMULA_IDS + (
    "tarantula", "ochiai", "ochiai2",
)
LEGACY14_FORMULA_IDS: Tuple[str, ...] = (
    "tarantula", "barinel", "kulczynski2", "wong2", "ochiai2", "ochiai",
    "dstar2", "jaccard", "wong3", "gp13", "op2", "wong1", "zoltar", "ample",
)
ALL_FORMULA_IDS: Tuple[str, ...] = TABLE4_FORMULA_IDS + tuple(
    formula_id for formula_id in LEGACY14_FORMULA_IDS
    if formula_id not in TABLE4_FORMULA_IDS
)


def formula_ids_for_set(name: str) -> Tuple[str, ...]:
    normalized = name.strip().lower()
    choices = {
        "table4": TABLE4_FORMULA_IDS,
        "study": STUDY_FORMULA_IDS,
        "legacy14": LEGACY14_FORMULA_IDS,
        "all": ALL_FORMULA_IDS,
    }
    if normalized not in choices:
        raise ValueError("formula set must be one of: table4, study, legacy14, all")
    return choices[normalized]


def evaluate_formula(spec: FormulaSpec, spectrum: Spectrum) -> FormulaEvaluation:
    """Evaluate one formula and encode every exceptional value explicitly."""

    spectrum.validate_training_domain()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = np.asarray(spec.evaluator(spectrum), dtype=np.float64)
    if raw.shape != spectrum.ef.shape:
        raw = np.broadcast_to(raw, spectrum.ef.shape).astype(np.float64, copy=True)
    else:
        raw = raw.copy()

    undefined = np.isnan(raw)
    positive = np.isposinf(raw)
    negative = np.isneginf(raw)
    finite = np.isfinite(raw)
    scores = raw.copy()
    scores[undefined] = spec.undefined_value

    finite_values = scores[np.isfinite(scores)]
    if positive.any():
        top = float(np.max(finite_values)) if finite_values.size else 0.0
        scores[positive] = top + max(1.0, abs(top))
    if negative.any():
        finite_values = scores[np.isfinite(scores)]
        bottom = float(np.min(finite_values)) if finite_values.size else 0.0
        scores[negative] = bottom - max(1.0, abs(bottom))

    if not np.isfinite(scores).all():
        raise ValueError("formula normalization failed for " + spec.formula_id)
    statuses = np.full(scores.shape, "finite", dtype="U24")
    statuses[undefined] = "undefined_neutral"
    statuses[positive] = "positive_infinity"
    statuses[negative] = "negative_infinity"
    return FormulaEvaluation(
        scores=scores,
        statuses=statuses,
        undefined_count=int(undefined.sum()),
        positive_infinity_count=int(positive.sum()),
        negative_infinity_count=int(negative.sum()),
    )


def evaluate_formula_set(
    formula_ids: Iterable[str], spectrum: Spectrum
) -> Dict[str, FormulaEvaluation]:
    selected = tuple(formula_ids)
    if not selected:
        raise ValueError("at least one formula is required")
    unknown = sorted(set(selected) - set(FORMULA_REGISTRY))
    if unknown:
        raise ValueError("unknown formula identifiers: " + ", ".join(unknown))
    if len(set(selected)) != len(selected):
        raise ValueError("formula identifiers cannot be duplicated")
    return {
        formula_id: evaluate_formula(FORMULA_REGISTRY[formula_id], spectrum)
        for formula_id in selected
    }


def formula_catalog_rows(formula_ids: Iterable[str]) -> Sequence[Dict[str, str]]:
    rows = []
    for formula_id in formula_ids:
        spec = FORMULA_REGISTRY[formula_id]
        rows.append({
            "formula_id": spec.formula_id,
            "name": spec.name,
            "expression": spec.expression,
            "families": ",".join(spec.families),
            "undefined_policy": "neutral_" + str(spec.undefined_value),
            "source": "Wong et al. 2016 Table 4" if "table4" in spec.families else "legacy repository",
        })
    return rows
