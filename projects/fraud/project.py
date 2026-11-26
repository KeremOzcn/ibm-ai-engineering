"""Evaluate train-only preprocessing and validation thresholds on later transactions."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from projects.common import file_sha256, runtime, source_identity, write_json


def chronological_splits(frame: pd.DataFrame) -> tuple[dict, dict]:
    """Remove repeated events and split at time boundaries, preserving natural prevalence."""
    required = {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}
    if not required.issubset(frame.columns) or not set(frame["Class"]).issubset({0, 1}):
        raise ValueError("Expected the credit-card dataset's numeric features and binary labels.")
    if not np.isfinite(frame[list(required)].to_numpy()).all():
        raise ValueError("Missing or non-finite transaction values.")
    data = frame.copy()
    data["record_id"] = np.arange(len(data))
    event_columns = sorted(required - {"Class"})
    conflicting = data.groupby(event_columns, dropna=False)["Class"].transform("nunique") > 1
    conflicts = data.loc[conflicting, "record_id"].tolist()
    data = data.loc[~conflicting]
    duplicates = data.duplicated(event_columns)
    repeated = data.loc[duplicates, "record_id"].tolist()
    data = data.loc[~duplicates].sort_values(["Time", "record_id"], kind="stable")
    cut1, cut2 = data["Time"].quantile([0.6, 0.8]).tolist()
    parts = {
        "train": data[data.Time < cut1],
        "validation": data[(data.Time >= cut1) & (data.Time < cut2)],
        "test": data[data.Time >= cut2],
    }
    if any(set(rows.Class) != {0, 1} for rows in parts.values()):
        raise ValueError("Every chronological partition must contain both labels.")
    audit = {
        "conflicting_ids": conflicts,
        "repeated_event_ids": repeated,
        "time_cuts": [cut1, cut2],
    }
    return parts, audit


def threshold_for_budget(scores, budget: int) -> float:
    """Choose a validation-only cutoff; ties must never exceed its review budget."""
    scores = np.asarray(scores, dtype=float)
    if budget < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("A positive budget and finite nonempty scores are required.")
    if budget >= len(scores):
        return float(scores.min())
    boundary = float(np.sort(scores)[-budget])
    return (
        float(np.nextafter(boundary, np.inf)) if (scores >= boundary).sum() > budget else boundary
    )


def alert_metrics(labels, predicted) -> dict:
    """Describe actual alerts and an explicitly illustrative unit-cost scenario."""
    labels, predicted = np.asarray(labels), np.asarray(predicted, dtype=bool)
    tp = int(((labels == 1) & predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    return {
        "alerts": int(predicted.sum()),
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_fraud": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "illustrative_cost": int(predicted.sum()) + 100 * fn,
    }


def run(data_file: Path, output: Path, *, seed: int = 42, budget: int = 100) -> dict:
    """Train fresh baselines, freeze validation cutoffs and evaluate the complete later test set."""
    import joblib
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    started = time.perf_counter()
    parts, exclusions = chronological_splits(pd.read_csv(data_file))
    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    manifest = {role: rows.record_id.astype(int).tolist() for role, rows in parts.items()}
    write_json(output / "split-manifest.json", {"partitions": manifest, "exclusions": exclusions})
    train, validation, test = [parts[role] for role in ["train", "validation", "test"]]
    factories = {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=seed
        ),
    }
    results = {}
    for name, model in factories.items():
        model.fit(train[features], train.Class)
        validation_scores = model.predict_proba(validation[features])[:, 1]
        cutoff = threshold_for_budget(validation_scores, budget)
        scores = model.predict_proba(test[features])[:, 1]
        selected = scores >= cutoff
        ranked = np.zeros(len(test), dtype=bool)
        ranked[np.argsort(-scores, kind="stable")[: min(budget, len(test))]] = True
        precision, recall, _ = precision_recall_curve(test.Class, scores)
        keep = np.unique(np.linspace(0, len(precision) - 1, min(256, len(precision)), dtype=int))
        path = output / f"{name}.joblib"
        joblib.dump(model, path)
        reloaded = joblib.load(path)  # Only the model created directly above is deserialized.
        np.testing.assert_array_equal(
            model.predict_proba(test[features].iloc[:16]),
            reloaded.predict_proba(test[features].iloc[:16]),
        )
        results[name] = {
            "average_precision": float(average_precision_score(test.Class, scores)),
            "roc_auc": float(roc_auc_score(test.Class, scores)),
            "validation_cutoff": cutoff,
            "validation_alerts": int((validation_scores >= cutoff).sum()),
            "fixed_threshold_test": alert_metrics(test.Class, selected),
            "top_budget_test": alert_metrics(test.Class, ranked),
            "test_samples": len(test),
            "exact_reload": True,
            "checkpoint_sha256": file_sha256(path),
            "pr_curve": {"precision": precision[keep].tolist(), "recall": recall[keep].tolist()},
        }
    report = {
        "project": "fraud",
        "profile": "chronological-complete-splits",
        "seed": seed,
        "data_sha256": file_sha256(data_file),
        "split_manifest_sha256": file_sha256(output / "split-manifest.json"),
        "split_counts": {
            role: {
                "samples": len(rows),
                "fraud": int(rows.Class.sum()),
                "min_time": float(rows.Time.min()),
                "max_time": float(rows.Time.max()),
            }
            for role, rows in parts.items()
        },
        "review_budget": budget,
        "features": features,
        "models": results,
        "threshold_policy": f"Validation-only fixed cutoff; future alert counts can drift. Separately report a fixed top-{budget} batch ranking.",
        "cost_assumptions": "Illustration only: one unit per reviewed alert and 100 units per missed fraud; no production cost estimate.",
        "limitations": "Two-day historical anonymized dataset; unknown customer grouping and upstream PCA fitting provenance. No deployment or realized financial benefit claimed.",
        "seconds": time.perf_counter() - started,
        "runtime": runtime(["numpy", "pandas", "scikit-learn", "joblib"]),
        "source": source_identity(
            globals().get("__file__"), globals().get("PROJECT_SOURCE_SHA256")
        ),
    }
    write_json(output / "metrics.json", report)
    return report
