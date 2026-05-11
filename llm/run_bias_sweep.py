import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from gpt_llm import predict_admission, load_model_and_encodings

SWEEPS = {
    "race": ["asian", "black or african american", "hispanic or latino",
             "other/underrepresented", "white"],
    "gender": ["F", "M"],
    "household_income": [15000, 40000, 85000, 180000],
    "gpa": [3.0, 3.4, 3.7, 4.0],
    "act_score": [22, 26, 30, 34],
}


def run_sweep_for_row(row, model, encodings):
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
    s = (
        df_preds.groupby(["student_id", "vary_feature"])["admit_probability"]
        .agg(prob_min="min", prob_max="max", prob_mean="mean")
        .reset_index()
    )
    s["prob_range"] = s["prob_max"] - s["prob_min"]
    return s


def make_bias_chart(summary_df, preds_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    feature_order = ["race", "household_income", "gender", "gpa", "act_score"]
    data = [summary_df.loc[summary_df["vary_feature"] == f, "prob_range"].values
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

    race_rows = preds_df[preds_df["vary_feature"] == "race"]
    race_means = race_rows.groupby("swept_value")["admit_probability"].mean().sort_values()

    gender_rows = preds_df[preds_df["vary_feature"] == "gender"]
    gender_means = gender_rows.groupby("swept_value")["admit_probability"].mean()

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
    ap.add_argument("--input", default=os.path.join(_HERE, "students.xlsx"))
    ap.add_argument("--predictions", default=os.path.join(_HERE, "predictions.csv"))
    ap.add_argument("--bias-summary", default=os.path.join(_HERE, "bias_summary.csv"))
    ap.add_argument("--chart", default=os.path.join(_HERE, "bias_chart.png"))
    args = ap.parse_args()

    model, encodings = load_model_and_encodings()
    df_students = pd.read_excel(args.input, sheet_name="students")
    print(f"{len(df_students)} students")

    all_rows = []
    for idx, row in df_students.iterrows():
        all_rows.extend(run_sweep_for_row(row, model, encodings))

    df_preds = pd.DataFrame(all_rows)
    df_preds.to_csv(args.predictions, index=False)

    summary_df = compute_bias_summary(df_preds)
    summary_df.to_csv(args.bias_summary, index=False)

    make_bias_chart(summary_df, df_preds, args.chart)

    print("\nbias summary by feature (mean within-student range)")
    headline = summary_df.groupby("vary_feature")["prob_range"].agg(["mean", "max", "count"])
    headline = headline.sort_values("mean", ascending=False)
    print(headline.to_string())


if __name__ == "__main__":
    main()