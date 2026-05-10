"""
run_bias_sweep.py -- batch runner for the chance-me random forest.

Reads students.xlsx (50 baseline profiles, one `vary_feature` per row),
runs a counterfactual sweep on the named feature while holding everything
else fixed, writes predictions.csv, and saves bias_chart.png.

Why counterfactuals instead of just running 50 LLM chats:
  The random forest is deterministic given inputs, so the only way to
  isolate bias on (say) race is to change ONLY race for the same student
  and see how the predicted admit probability moves. That's the
  individual-fairness / counterfactual-fairness lens.

The bias chart shows, for each feature swept, the distribution of
within-student probability ranges (max - min across the sweep). Large
ranges = the model is highly sensitive to that single attribute even
with everything else held constant.

Usage:
    python run_bias_sweep.py
        --input students.xlsx
        --predictions predictions.csv
        --chart bias_chart.png
"""
import argparse
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the exact prediction function from the LLM script so we know we're
# scoring identically.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_llm import predict_admission, load_model_and_encodings  # noqa: E402


# Sweep grids -- the set of counterfactual values we try for each feature.
# Race and gender sweep the full category list; numeric features sweep a
# small representative ladder.
SWEEPS = {
    "race": ["asian", "black or african american", "hispanic or latino",
             "other/underrepresented", "white"],
    "gender": ["F", "M"],
    "household_income": [15000, 40000, 85000, 180000],
    "gpa": [3.0, 3.4, 3.7, 4.0],
    "act_score": [22, 26, 30, 34],
}


def run_sweep_for_row(row, model, encodings):
    """Run baseline + counterfactuals for one student row.

    Returns a list of dicts, one per scored profile.
    """
    feature = row["vary_feature"]
    if feature not in SWEEPS:
        print(f"  [warn] unknown vary_feature '{feature}' for {row['student_id']}, skipping")
        return []

    base = {
        "gpa": float(row["gpa"]),
        "act_score": float(row["act_score"]),
        "household_income": float(row["household_income"]),
        "gender": str(row["gender"]),
        "race": str(row["race"]),
    }

    results = []
    for swept_value in SWEEPS[feature]:
        profile = dict(base)
        profile[feature] = swept_value
        pred = predict_admission(profile, model, encodings)
        if "error" in pred:
            print(f"  [error] {row['student_id']} {feature}={swept_value}: {pred['error']}")
            continue
        results.append({
            "student_id": row["student_id"],
            "vary_feature": feature,
            "swept_value": swept_value,
            "is_baseline": profile[feature] == base[feature],
            "gpa": profile["gpa"],
            "act_score": profile["act_score"],
            "household_income": profile["household_income"],
            "gender": profile["gender"],
            "race": profile["race"],
            "admit_probability": pred["admit_probability"],
        })
    return results


def compute_bias_summary(df_preds):
    """For each (student, vary_feature), compute max-min admit prob across sweep."""
    summary = (
        df_preds.groupby(["student_id", "vary_feature"])["admit_probability"]
        .agg(prob_min="min", prob_max="max", prob_mean="mean")
        .reset_index()
    )
    summary["prob_range"] = summary["prob_max"] - summary["prob_min"]
    return summary


def make_bias_chart(summary, out_path):
    """Two-panel chart:

    Left:  boxplot of within-student probability ranges, by feature swept.
           A box sitting high = the model swings a lot when ONLY that
           feature changes = more bias on that axis.
    Right: mean admit probability by category, for race and gender
           (averaged across all students who had that feature swept).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Panel 1: range by feature -------------------------------------
    feature_order = ["race", "household_income", "gender", "gpa", "act_score"]
    data = [summary.loc[summary["vary_feature"] == f, "prob_range"].values
            for f in feature_order]

    bp = axes[0].boxplot(data, tick_labels=feature_order, patch_artist=True,
                         medianprops=dict(color="black", linewidth=1.5))
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c", "#9370db"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    axes[0].set_ylabel("Within-student probability range\n(max − min across sweep)")
    axes[0].set_title("How much does ONE feature swing the prediction?\n"
                      "(higher = more bias on that axis)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_ylim(bottom=0)

    # ---- Panel 2: mean admit prob by race / gender ---------------------
    # Pull the raw prediction rows so we can group by the swept value.
    # We attach it via attribute for convenience.
    df = make_bias_chart._preds_df

    race_rows = df[df["vary_feature"] == "race"]
    race_means = race_rows.groupby("swept_value")["admit_probability"].mean()
    race_means = race_means.sort_values()

    gender_rows = df[df["vary_feature"] == "gender"]
    gender_means = gender_rows.groupby("swept_value")["admit_probability"].mean()

    # Plot race as horizontal bars, gender as a small overlay on the right.
    ypos = np.arange(len(race_means))
    bars = axes[1].barh(ypos, race_means.values, color="#5bc0de",
                       edgecolor="black", alpha=0.85)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(race_means.index)
    axes[1].set_xlabel("Mean predicted admit probability")
    axes[1].set_title("Mean admit probability when ONLY race is swept\n"
                     "(same students, identical GPA/ACT/income/gender)")
    axes[1].set_xlim(0, max(race_means.max() * 1.25, 0.1))
    for bar, val in zip(bars, race_means.values):
        axes[1].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    # Gender means as text annotation
    g_text = "Gender sweep means: " + ", ".join(
        f"{k}={v:.3f}" for k, v in gender_means.items()
    )
    axes[1].text(0.0, -0.18, g_text, transform=axes[1].transAxes,
                fontsize=9, style="italic")

    fig.suptitle("Counterfactual bias sweep -- chance-me random forest",
                fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="students.xlsx")
    ap.add_argument("--predictions", default="predictions.csv")
    ap.add_argument("--bias-summary", default="bias_summary.csv")
    ap.add_argument("--chart", default="bias_chart.png")
    args = ap.parse_args()

    print(f"Loading model + encodings ...")
    model, encodings = load_model_and_encodings()

    print(f"Reading {args.input} ...")
    df_students = pd.read_excel(args.input, sheet_name="students")
    print(f"  -> {len(df_students)} students")

    all_rows = []
    for idx, row in df_students.iterrows():
        print(f"[{idx + 1:>2}/{len(df_students)}] {row['student_id']} "
              f"(vary={row['vary_feature']}) ...")
        all_rows.extend(run_sweep_for_row(row, model, encodings))

    df_preds = pd.DataFrame(all_rows)
    df_preds.to_csv(args.predictions, index=False)
    print(f"\nwrote {args.predictions}  ({len(df_preds)} prediction rows)")

    summary = compute_bias_summary(df_preds)
    summary.to_csv(args.bias_summary, index=False)
    print(f"wrote {args.bias_summary}  ({len(summary)} student-feature rows)")

    make_bias_chart._preds_df = df_preds
    make_bias_chart(summary, args.chart)
    print(f"wrote {args.chart}")

    # Quick console summary so you can see headline numbers right away.
    print("\n=== bias summary by feature (mean within-student range) ===")
    headline = summary.groupby("vary_feature")["prob_range"].agg(["mean", "max", "count"])
    headline = headline.sort_values("mean", ascending=False)
    print(headline.to_string())


if __name__ == "__main__":
    main()
