"""Charts (PNG, ready for slides) and a plain-language summary with your real numbers.

Run from the project folder (after modules.train):   python -m modules.report
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

LABELS = {"persistence": "Yesterday's output", "physics": "Formula (0.8 ratio)",
          "physics_calibrated": "Formula (fitted ratio)", "random_forest": "Random Forest (AI)"}
COLORS = {"persistence": "#8a8f98", "physics": "#b8bcc4", "physics_calibrated": "#6b7280",
          "random_forest": "#1f6f3b"}


def _save(fig, name):
    Path(config.FIG_FOLDER).mkdir(parents=True, exist_ok=True)
    path = Path(config.FIG_FOLDER) / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_irradiance_vs_kwh(table):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(table["forecast_irr_kwh_m2"], table["kwh"], s=14, alpha=0.6, color="#1f6f3b")
    ax.set_xlabel("Forecast sunlight (kWh/m2 per day)")
    ax.set_ylabel("Measured output (kWh per day)")
    ax.set_title("More forecast sunlight, more energy (NREL system 1332)")
    return _save(fig, "1_sunlight_vs_output.png")


def chart_test_days(pred):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(pred.index, pred["kwh"], color="black", lw=2, label="Actual")
    for m in ("persistence", "physics_calibrated", "random_forest"):
        ax.plot(pred.index, pred[m], color=COLORS[m], lw=1.3, label=LABELS[m],
                ls="-" if m == "random_forest" else "--")
    ax.set_ylabel("kWh per day")
    ax.set_title("Test days the model never saw: actual against forecasts")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)       # headroom so the legend sits clear of the lines
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    fig.autofmt_xdate()
    return _save(fig, "2_test_days.png")


def chart_mae_bars(ev):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = list(LABELS)
    vals = [ev.loc[m, "MAE_kWh"] for m in order]
    bars = ax.bar([LABELS[m] for m in order], vals, color=[COLORS[m] for m in order])
    ax.bar_label(bars, fmt="%.0f", fontsize=9)
    ax.set_ylabel("Average error (kWh per day), lower is better")
    ax.set_title("How far off is each method?")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, "3_error_by_method.png")


def chart_pred_vs_actual(pred):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = [0, max(pred["kwh"].max(), pred["random_forest"].max()) * 1.05]
    ax.plot(lim, lim, color="gray", lw=1, ls="--", label="Perfect")
    ax.scatter(pred["kwh"], pred["random_forest"], s=18, color=COLORS["random_forest"])
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual (kWh per day)"); ax.set_ylabel("Predicted (kWh per day)")
    ax.set_title("Random Forest: predicted against actual"); ax.legend()
    return _save(fig, "4_predicted_vs_actual.png")


def _skill_sentence(name, skill, mae_b):
    pct = abs(skill) * 100
    if skill > 0:
        return f"- Against **{name}** (error {mae_b:.0f} kWh): the AI makes **{pct:.0f}% fewer errors** (skill {skill:+.2f})."
    return (f"- Against **{name}** (error {mae_b:.0f} kWh): the AI does **not** beat it "
            f"(skill {skill:+.2f}). Report this honestly; it is still a result.")


def write_summary(ev, skill, imp, info):
    avg = info["average_test_kwh"]
    mae_rf = ev.loc["random_forest", "MAE_kWh"]
    rmse_rf = ev.loc["random_forest", "RMSE_kWh"]
    pct_rf = ev.loc["random_forest", "MAE_pct_of_avg"]
    top = imp.head(3)
    lines = [
        "# Results summary (written from your real numbers)", "",
        "## The data",
        f"- {info['train_days'] + info['test_days']} scored days: {info['train_days']} for training "
        f"({info['train_period'][0]} to {info['train_period'][1]}) and {info['test_days']} for testing "
        f"({info['test_period'][0]} to {info['test_period'][1]}), split by date and never shuffled.",
        f"- The system makes about {avg:.0f} kWh on an average test day.", "",
        "## Results on the test days (days the model never saw)", "",
        "| Method | MAE (kWh/day) | RMSE (kWh/day) | MAE as % of average |", "|---|---|---|---|",
    ]
    for m in LABELS:
        r = ev.loc[m]
        lines.append(f"| {LABELS[m]} | {r['MAE_kWh']:.1f} | {r['RMSE_kWh']:.1f} | {r['MAE_pct_of_avg']:.1f}% |")
    lines += ["", "## Skill: does the AI beat the simple methods?", ""]
    for b in ("persistence", "physics", "physics_calibrated"):
        lines.append(_skill_sentence(LABELS[b], skill.loc[b, "skill"], skill.loc[b, "baseline_MAE_kWh"]))
    lines += [
        "", "## Say it in plain words", "",
        f"> \"On days the model had never seen, our forecast was off by {mae_rf:.0f} kWh on average, "
        f"about {pct_rf:.0f}% of a normal day's output ({avg:.0f} kWh).\"", "",
        "## What the model relies on most", "",
    ] + [f"- {r.feature}: {r.importance:.2f}" for r in top.itertuples()]
    lines += [
        "", f"Input set used: **{info['feature_set'].replace('_', ' ')}** (chosen using training days only). "
        f"On the test days the other input set scored {info['test_mae_other_feature_set']:.0f} kWh, shown for transparency.",
        "", "## Limits to state", "",
        "- About one year of data, so each season is seen once; the test days fall in autumn and winter.",
        "- Weather inputs are archived forecast model outputs, not forecasts issued exactly 24 hours ahead.",
        "- One system in Colorado. This proves the method, not that it works in Jamaica.",
        f"- RMSE ({rmse_rf:.0f} kWh) above MAE ({mae_rf:.0f} kWh) means a few days are missed by much more than average.",
    ]
    Path(config.SUMMARY_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(config.SUMMARY_FILE).write_text("\n".join(lines) + "\n")


def main():
    ev = pd.read_csv(config.EVAL_ALL_FILE, index_col=0)
    skill = pd.read_csv(config.SKILL_FILE, index_col="baseline")
    imp = pd.read_csv(config.IMPORTANCE_FILE)
    pred = pd.read_csv(config.PRED_FILE, index_col="date", parse_dates=True)
    table = pd.read_csv(config.JOINED_FILE, index_col="date", parse_dates=True)
    info = json.loads(Path(config.MODEL_INFO_FILE).read_text())

    paths = [chart_irradiance_vs_kwh(table), chart_test_days(pred),
             chart_mae_bars(ev), chart_pred_vs_actual(pred)]
    write_summary(ev, skill, imp, info)
    print("Charts saved:")
    for p in paths:
        print("  ", p)
    print("Summary saved:", config.SUMMARY_FILE)


if __name__ == "__main__":
    main()
