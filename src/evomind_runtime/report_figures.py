"""Deterministic figure renderer. Source arrays and provenance travel with plots."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


def numeric(values, *, missing=False):
    if not isinstance(values, list) or not values or len(values) > 20000:
        raise ValueError("figure_array_invalid")
    result = []
    for value in values:
        if value is None and missing:
            result.append(float("nan"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result.append(float(value))
        else:
            raise ValueError("figure_value_invalid")
    return result


def roc_points(labels, scores):
    y, score = numeric(labels), numeric(scores)
    if len(y) != len(score) or set(y) != {0.0, 1.0}:
        raise ValueError("binary_prediction_source_required")
    ordered = sorted(zip(score, y), key=lambda item: -item[0])
    positives, negatives = sum(y), len(y) - sum(y)
    tp = fp = 0
    points = [(0.0, 0.0)]
    for index, (threshold, target) in enumerate(ordered):
        tp += target == 1
        fp += target == 0
        if index + 1 == len(ordered) or ordered[index + 1][0] != threshold:
            points.append((fp / negatives, tp / positives))
    return points


def pr_points(labels, scores):
    y, score = numeric(labels), numeric(scores)
    if len(y) != len(score) or set(y) != {0.0, 1.0}:
        raise ValueError("binary_prediction_source_required")
    ordered = sorted(zip(score, y), key=lambda item: -item[0])
    positives = sum(y)
    tp = 0
    points = [(0.0, 1.0)]
    for index, (threshold, target) in enumerate(ordered):
        tp += target == 1
        if index + 1 == len(ordered) or ordered[index + 1][0] != threshold:
            points.append((tp / positives, tp / (index + 1)))
    return points


def render_figures(specs, output):
    if not specs:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for candidate in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            plt.rcParams["font.family"] = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "evomind-report-v1", "axes.unicode_minus": False, "text.usetex": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, source in enumerate(specs):
        identifier = f"figure-{index+1:02d}"
        figure = None
        row = {"id": identifier, "title": str(source.get("title") or identifier),
               "caption": str(source.get("caption") or "Recorded source data; not an independent scientific claim."),
               "source_artifact_id": source.get("source_artifact_id"), "source_sha256": source.get("source_sha256")}
        try:
            if not re.fullmatch(r"[a-f0-9]{64}", str(source.get("source_sha256") or "")) or not source.get("source_artifact_id"):
                raise ValueError("figure_source_identity_missing")
            figure, ax = plt.subplots(figsize=(7.1, 4.3), layout="constrained")
            kind = source.get("kind")
            table = []
            if kind in {"line", "scatter"}:
                x, y = numeric(source.get("x")), numeric(source.get("y"), missing=True)
                if len(x) != len(y):
                    raise ValueError("figure_length_mismatch")
                if kind == "scatter" or source.get("points_only"):
                    ax.scatter(x, y, s=18, color="#147b70", alpha=0.85)
                else:
                    ax.plot(x, y, color="#147b70", linewidth=1.7, marker="." if len(x) < 100 else None)
                table = [(a, "" if math.isnan(b) else b) for a, b in zip(x, y)]
            elif kind == "bar":
                labels, y = source.get("labels"), numeric(source.get("y"))
                if not isinstance(labels, list) or len(labels) != len(y) or len(y) > 40:
                    raise ValueError("figure_labels_invalid")
                ax.bar([str(value) for value in labels], y, color="#147b70")
                table = list(zip(labels, y))
            elif kind == "roc":
                table = roc_points(source.get("y_true"), source.get("y_score"))
                ax.plot([x for x, _y in table], [y for _x, y in table], color="#147b70")
                ax.plot([0, 1], [0, 1], linestyle="--", color="#849591", linewidth=1)
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
            elif kind == "pr":
                table = pr_points(source.get("y_true"), source.get("y_score"))
                ax.step([x for x, _y in table], [y for _x, y in table], where="post", color="#147b70")
                ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
            elif kind in {"regression", "residual"}:
                observed, predicted = numeric(source.get("y_true")), numeric(source.get("y_pred"))
                if len(observed) != len(predicted): raise ValueError("prediction_length_mismatch")
                if kind == "regression":
                    table = list(zip(observed, predicted))
                    ax.scatter(observed, predicted, s=18, alpha=0.8, color="#147b70")
                    ends = [min(observed + predicted), max(observed + predicted)]
                    ax.plot(ends, ends, linestyle="--", color="#849591", linewidth=1)
                    ax.set(xlabel="Observed", ylabel="Predicted")
                else:
                    residuals = [prediction - truth for truth, prediction in zip(observed, predicted)]
                    table = list(zip(predicted, residuals))
                    ax.scatter(predicted, residuals, s=18, alpha=0.8, color="#147b70")
                    ax.axhline(0, linestyle="--", color="#849591", linewidth=1)
                    ax.set(xlabel="Predicted", ylabel="Residual (predicted - observed)")
            elif kind == "confusion_matrix":
                matrix = source.get("matrix")
                if not isinstance(matrix, list) or not 1 <= len(matrix) <= 30:
                    raise ValueError("confusion_matrix_invalid")
                matrix = [numeric(row) for row in matrix]
                if any(len(row) != len(matrix) or any(value < 0 for value in row) for row in matrix):
                    raise ValueError("confusion_matrix_invalid")
                heatmap = ax.imshow(matrix, cmap="GnBu")
                figure.colorbar(heatmap, ax=ax, label="Count")
                maximum = max(max(values) for values in matrix)
                for y_index, values in enumerate(matrix):
                    for x_index, value in enumerate(values):
                        if len(matrix) <= 10:
                            ax.text(x_index, y_index, f"{value:g}", ha="center", va="center", color="white" if value > maximum * 0.55 else "#102823")
                ax.set(xlabel="Predicted class", ylabel="Observed class")
                table = [(i, j, value) for i, values in enumerate(matrix) for j, value in enumerate(values)]
            elif kind == "comparison":
                baseline, candidate = source.get("baseline", {}), source.get("candidate", {})
                for key in ("protocol_sha256", "data_sha256", "split_sha256", "scorer_sha256"):
                    if not re.fullmatch(r"[a-f0-9]{64}", str(baseline.get(key) or "")) or baseline.get(key) != candidate.get(key):
                        raise ValueError("comparison_not_comparable")
                if baseline.get("metric") != candidate.get("metric") or baseline.get("direction") not in {"higher", "lower"} or baseline.get("direction") != candidate.get("direction"):
                    raise ValueError("comparison_not_comparable")
                y = numeric([baseline.get("value"), candidate.get("value")])
                labels = [str(baseline.get("label") or "Baseline"), str(candidate.get("label") or "Candidate")]
                ax.bar(labels, y, color=["#849591", "#147b70"])
                table = list(zip(labels, y))
                row["direction"] = baseline["direction"]
            else:
                raise ValueError("figure_kind_unsupported")
            ax.set_title(row["title"], loc="left", fontweight="semibold", pad=12)
            if source.get("x_label"): ax.set_xlabel(str(source["x_label"]))
            if source.get("y_label"): ax.set_ylabel(str(source["y_label"]))
            if kind != "confusion_matrix": ax.grid(axis="y", alpha=0.2)
            for extension in ("svg", "png"):
                target = output / f"{identifier}.{extension}"
                if target.exists(): raise ValueError("figure_output_exists")
                figure.savefig(target, dpi=300, facecolor="white", metadata={"Date": None} if extension == "svg" else {})
                row[extension] = target.name
            with (output / f"{identifier}.csv").open("x", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row", "column", "value"] if kind == "confusion_matrix" else ["x", "y"])
                writer.writerows(table)
            (output / f"{identifier}.source.json").write_text(json.dumps(source, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
            row.update(status="ready", csv=f"{identifier}.csv", recipe=f"{identifier}.source.json")
        except (ValueError, TypeError, KeyError) as error:
            row.update(status="missing_data", reason=str(error))
        finally:
            if figure is not None: plt.close(figure)
        results.append(row)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    print(json.dumps(render_figures(source if isinstance(source, list) else [source], Path(args.out_dir))))
