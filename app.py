from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from modules import jamaica

st.set_page_config(page_title="Solar Predict JA", layout="wide")

st.warning(
    f"Research prototype. Measured data: NREL PVDAQ system {config.SYSTEM_ID} "
    f"({config.SYSTEM_NAME}, {config.SIZE_KW} kW). The Jamaica tab is a simulation and is not validated."
)
st.title("Solar Predict JA")
st.caption("Day-ahead solar generation forecasting (kWh) for utility generation planning.")


@st.cache_data
def load_csv(path, index_col=None, parse_dates=False):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=index_col, parse_dates=parse_dates)


preds_df = load_csv(config.PRED_FILE, index_col="date", parse_dates=True)
joined_df = load_csv(config.JOINED_FILE, index_col="date", parse_dates=True)
eval_df = load_csv(config.EVAL_ALL_FILE, index_col=0)
skill_df = load_csv(config.SKILL_FILE, index_col=0)
imp_df = load_csv(config.IMPORTANCE_FILE)

LABELS = {
    "random_forest": "Random Forest",
    "persistence": "Yesterday's output (persistence)",
    "physics": "Physics formula (0.8)",
    "physics_calibrated": "Physics formula (fitted ratio)",
}
RUN_FIRST = "Run `python -m modules.prepare` then `python -m modules.train` first to create the results files."

tomorrow, past, check, ja_tab = st.tabs(
    ["Tomorrow", "Past performance", "Model check", "Jamaica demo"]
)

# ======================= TAB 1: TOMORROW =======================
with tomorrow:
    st.subheader("Day-ahead forecast (replay of a held-out test day)")
    st.caption(
        "Each date below was never used for training. The inputs are what would have been "
        "known the day before: the weather forecast and earlier days' output."
    )
    if preds_df is None or eval_df is None:
        st.info(RUN_FIRST)
    else:
        min_d, max_d = preds_df.index.min().date(), preds_df.index.max().date()
        chosen = st.date_input("Choose a test day:", value=max_d, min_value=min_d, max_value=max_d)
        chosen_ts = pd.Timestamp(chosen)
        if chosen_ts not in preds_df.index:
            st.warning("No complete data for that date (removed by the 90% completeness rule).")
        else:
            row = preds_df.loc[chosen_ts]
            rf_mae = float(eval_df.loc["random_forest", "MAE_kWh"])
            c1, c2 = st.columns(2)
            c1.metric(
                "Forecast (Random Forest)",
                f"{row['random_forest']:,.0f} kWh",
                f"typical error about {rf_mae:,.0f} kWh",
                delta_color="off",
            )
            c2.metric("Actual measured output", f"{row['kwh']:,.0f} kWh")

            st.markdown("#### The same day, other methods")
            others = [m for m in ("persistence", "physics", "physics_calibrated") if m in preds_df.columns]
            cols = st.columns(len(others))
            for col, m in zip(cols, others):
                col.metric(LABELS[m], f"{row[m]:,.0f} kWh")

            wx = None
            if joined_df is not None and chosen_ts in joined_df.index:
                wx = joined_df.loc[chosen_ts]
            if wx is not None:
                st.markdown("#### Weather forecast inputs (issued the day before)")
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Forecast sunlight", f"{wx['forecast_irr_kwh_m2']:.2f} kWh/m²")
                w2.metric("Forecast cloud cover", f"{wx['forecast_cloud_pct']:.0f}%")
                w3.metric("Forecast temperature", f"{wx['forecast_temp_c']:.1f} °C")
                if "clearsky_kwh_m2" in wx.index:
                    w4.metric("Clear-sky ceiling", f"{wx['clearsky_kwh_m2']:.2f} kWh/m²")
            st.caption("The 'typical error' is the model's average error on all test days. It is not a guaranteed range.")

# ======================= TAB 2: PAST PERFORMANCE =======================
with past:
    st.subheader("Actual vs forecast (held-out test period)")
    if preds_df is None:
        st.info(RUN_FIRST)
    else:
        months = sorted(preds_df.index.to_period("M").astype(str).unique())
        sel_month = st.selectbox("Choose a month:", months, index=len(months) - 1)
        sub = preds_df[preds_df.index.to_period("M").astype(str) == sel_month]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub.index, y=sub["kwh"], mode="lines+markers", name="Actual (measured)"))
        fig.add_trace(go.Scatter(x=sub.index, y=sub["random_forest"], mode="lines+markers", name=LABELS["random_forest"]))
        for m, dash in (("persistence", "dot"), ("physics", "dash"), ("physics_calibrated", "dashdot")):
            if m in sub.columns:
                fig.add_trace(go.Scatter(x=sub.index, y=sub[m], mode="lines", name=LABELS[m], line=dict(dash=dash)))
        fig.update_layout(
            title=f"Daily solar energy, {sel_month}",
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================= TAB 3: MODEL CHECK =======================
with check:
    st.subheader("Held-out test scores")
    if eval_df is None or preds_df is None:
        st.info(RUN_FIRST)
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Error (lower is better)**")
            st.dataframe(eval_df, use_container_width=True)
        with right:
            if skill_df is not None:
                st.markdown("**Skill against each baseline (above 0 = the model wins)**")
                st.dataframe(skill_df, use_container_width=True)
        st.caption(
            "Skill = 1 - model MAE / baseline MAE. The model also uses recent peak power as an input, "
            "because the plant's equipment changed during the year. Gains are small on a short test period, "
            "so read them as 'slightly better', not 'proven better'."
        )

        scatter_df = preds_df.reset_index()
        fig_s = px.scatter(
            scatter_df, x="kwh", y="random_forest", hover_data=["date"],
            labels={"kwh": "Actual measured (kWh)", "random_forest": "Forecast, Random Forest (kWh)"},
            title="Forecast vs actual, test days",
        )
        top = float(max(preds_df["kwh"].max(), preds_df["random_forest"].max()))
        fig_s.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(color="gray", dash="dash"))
        st.plotly_chart(fig_s, use_container_width=True)

        if imp_df is not None:
            st.markdown("**What the model leaned on (not the same as cause)**")
            st.bar_chart(imp_df.set_index("feature")["importance"])

# ======================= TAB 4: JAMAICA DEMO =======================
with ja_tab:
    st.warning(
        "Not validated: a fictional 20 MW AC solar farm in Clarendon, simulated from Jamaican weather. "
        "It illustrates the method and proves nothing about real Jamaican plant output."
    )
    st.subheader("Jamaica transfer demonstration (fictional 20 MW AC farm, Clarendon)")

    @st.cache_data
    def get_jamaica_demo():
        ghi = jamaica.fetch_daily_ghi(17.97, -77.24, "2023-04-01", "2024-03-31", forecast=True)
        return jamaica.simulate_daily_kwh(ghi, dc_kw=20_000 * 1.3, performance_ratio=config.PERFORMANCE_RATIO)

    try:
        ja_df = get_jamaica_demo()
        sim_cf = jamaica.capacity_factor(ja_df, ac_kw=20_000)
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated output, Apr 2023 to Mar 2024", f"{ja_df['simulated_kwh'].sum() / 1000:,.0f} MWh")
        m2.metric("Simulated capacity factor", f"{sim_cf:.1%}")
        m3.metric("OUR figure, Content Solar", "26.1% (unverified)")
        st.caption(
            "The simulation and the OUR figure are different things: one is a formula on weather, "
            "the other is reported metered output. Do not read the gap as an error until the OUR number is checked."
        )
        fig_ja = px.line(
            ja_df, x="date", y="simulated_kwh",
            labels={"date": "Date", "simulated_kwh": "Simulated daily output (kWh)"},
            title="Simulated daily output, fictional 20 MW farm",
        )
        st.plotly_chart(fig_ja, use_container_width=True)
    except Exception as e:
        st.error(f"Could not get Jamaica weather data: {e}")
