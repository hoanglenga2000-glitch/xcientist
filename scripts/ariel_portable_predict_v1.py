"""Inference-only Ariel Ridge bundle loader. No pickle, fitting, or label access."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def predict(model_path: Path, features_path: Path, output_path: Path) -> None:
    with np.load(model_path, allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    with np.load(features_path, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float64)
        feature_names = archive["feature_names"].astype(str)
        planet_id = archive["planet_id"].astype(str)
    if str(model["schema"].item()) != "evomind.ariel.portable_ridge.v1":
        raise ValueError("unsupported_model_schema")
    if x.ndim != 2 or x.shape[1] != len(feature_names) or len(x) != len(planet_id):
        raise ValueError("feature_shape_mismatch")
    if not np.array_equal(feature_names, model["feature_names"]):
        raise ValueError("feature_order_mismatch")
    if np.isinf(x).any():
        raise ValueError("infinite_features")
    statistics = model["imputer_statistics"]
    keep = model["imputer_keep"].astype(bool)
    indicators = model["imputer_indicators"].astype(int)
    if statistics.shape != (x.shape[1],) or keep.shape != statistics.shape:
        raise ValueError("imputer_shape_mismatch")
    missing = np.isnan(x)
    transformed = np.where(missing, statistics[None, :], x)[:, keep]
    if len(indicators):
        transformed = np.concatenate((transformed, missing[:, indicators].astype(float)), axis=1)
    if transformed.shape[1] != len(model["scaler_mean"]):
        raise ValueError("scaler_shape_mismatch")
    if not np.isfinite(transformed).all() or np.any(model["scaler_scale"] <= 0):
        raise ValueError("invalid_transformed_features")
    transformed = (transformed - model["scaler_mean"]) / model["scaler_scale"]
    mu = np.maximum(transformed @ model["ridge_coef"].T + model["ridge_intercept"], 1e-15)
    sigma = np.broadcast_to(model["sigma"], mu.shape).copy()
    if mu.shape != (len(x), 283) or not np.isfinite(mu).all():
        raise ValueError("invalid_model_predictions")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("invalid_model_sigma")
    if output_path.exists():
        raise FileExistsError("prediction_output_already_exists")
    np.savez_compressed(output_path, planet_id=planet_id, mu=mu, sigma=sigma,
                        target_columns=model["target_columns"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(args.model, args.features, args.output)
