"""
surrogates.py

Declarative registry of surrogate models that regress physics quantities from
the labelled dataset (datagen.py), plus generic train / evaluate / persist
helpers. Each surrogate is defined once as a `SurrogateSpec`; adding a new one
is a single registry entry — no new training script needed.

The specs mirror the notebook experiments:
    A      : T0                        -> Istar_A      (runaway current vs bath T)
    two_d  : T0, lam                   -> Istar_A      (adds cooling as input)
    B      : T0, I                     -> margin_norm  (protection-loop quantity)
    raw    : T0, I, voltage, T_surface -> Istar_A      (live-instrument inputs)

Usage
-----
    from hts_tape_env.datagen import generate_dataset
    from hts_tape_env.surrogates import train_spec, SPECS

    df = generate_dataset(5000, lam_range=(20, 200))   # 'two_d' needs lam
    result = train_spec("two_d", df)
    print(result.metrics)                              # {'mae':..., 'r2':...}
    result.save("models/surr_two_d.joblib")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(max_leaf_nodes: int = 63, **overrides) -> Pipeline:
    """StandardScaler -> HistGradientBoostingRegressor, matching the notebook."""
    params = dict(
        max_iter=800,
        learning_rate=0.04,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=0.05,
        validation_fraction=0.10,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )
    params.update(overrides)
    return make_pipeline(StandardScaler(), HistGradientBoostingRegressor(**params))


# ── Surrogate specification ───────────────────────────────────────────────────

@dataclass(frozen=True)
class SurrogateSpec:
    """One regression task: which columns are inputs, which is the target."""
    name: str
    features: tuple[str, ...]
    target: str
    dedup_on: tuple[str, ...] | None = None   # collapse duplicate rows (e.g. T0->I*)
    max_leaf_nodes: int = 63
    description: str = ""

    def prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Select feature/target arrays, dropping NaN rows and optional dedup."""
        cols = list(self.features) + [self.target]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"spec '{self.name}' needs columns {missing} not in dataset. "
                f"(Did you generate with the right options, e.g. lam_range for 'two_d'?)"
            )
        sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if self.dedup_on:
            sub = sub.drop_duplicates(subset=list(self.dedup_on))
        X = sub[list(self.features)].to_numpy(dtype=float)
        y = sub[self.target].to_numpy(dtype=float)
        return X, y


# Registry — the source of truth for "what surrogates exist".
SPECS: dict[str, SurrogateSpec] = {
    "A": SurrogateSpec(
        name="A",
        features=("T0_K",),
        target="Istar_A",
        dedup_on=("T0_K",),
        max_leaf_nodes=31,
        description="Runaway current vs bath temperature: T0 -> I*.",
    ),
    "two_d": SurrogateSpec(
        name="two_d",
        features=("T0_K", "lam"),
        target="Istar_A",
        description="Runaway current with explicit cooling: (T0, lambda) -> I*. "
                    "Wire into env.set_surrogate() to replace per-step brentq.",
    ),
    "B": SurrogateSpec(
        name="B",
        features=("T0_K", "I_A"),
        target="margin_norm",
        max_leaf_nodes=31,
        description="Protection-loop margin: (T0, I) -> (I*-I)/I*.",
    ),
    "raw": SurrogateSpec(
        name="raw",
        features=("T0_K", "I_A", "voltage_V", "T_surface_K"),
        target="Istar_A",
        description="Live-instrument inputs: (T0, I, V, T_surface) -> I*.",
    ),
}


# ── Train / evaluate / persist ────────────────────────────────────────────────

@dataclass
class TrainResult:
    spec: SurrogateSpec
    model: Pipeline
    metrics: dict
    n_train: int
    n_test: int
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Persist model + spec + metrics as a single joblib bundle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "spec": self.spec,
                "features": list(self.spec.features),
                "target": self.spec.target,
                "metrics": self.metrics,
                "meta": self.meta,
            },
            path,
        )
        return path


def train_spec(
    name: str,
    df: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    model_overrides: dict | None = None,
) -> TrainResult:
    """Train the named surrogate on `df` and return model + held-out metrics."""
    if name not in SPECS:
        raise KeyError(f"unknown surrogate '{name}'. Available: {list(SPECS)}")
    spec = SPECS[name]
    X, y = spec.prepare(df)
    if len(y) < 10:
        raise ValueError(f"spec '{name}': only {len(y)} usable rows after cleaning.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = make_model(max_leaf_nodes=spec.max_leaf_nodes, **(model_overrides or {}))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    denom = np.where(np.abs(y_te) > 1e-12, np.abs(y_te), np.nan)
    metrics = {
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "median_rel_err": float(np.nanmedian(np.abs(pred - y_te) / denom)),
    }
    return TrainResult(
        spec=spec, model=model, metrics=metrics,
        n_train=len(y_tr), n_test=len(y_te),
        meta={"features": list(spec.features), "target": spec.target},
    )


def load_surrogate(path: str | Path) -> dict:
    """Load a saved bundle: {'model', 'spec', 'features', 'target', 'metrics', 'meta'}."""
    return joblib.load(Path(path))
