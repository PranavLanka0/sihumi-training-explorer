"""
SIHUMI lactose-training: interactive multivariate analysis.

Streamlit front-end over NatiRios_multivariate_analysis.py. Runs locally
(`streamlit run app.py`) or on Streamlit Community Cloud.

Everything shown is recomputed from the table in the Data tab, so edits
propagate to every plot and statistic.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

import NatiRios_multivariate_analysis as mv

st.set_page_config(page_title="SIHUMI training - multivariate analysis",
                   layout="wide")

GREY, PINK = "#8A8A8A", "#E8696E"
LABELS = list(mv.LABELS)
SPECIES_ORDER = ["E. coli", "E. faecalis", "L. plantarum", "M. gnavus",
                 "B. longum", "F. duncaniae", "P. vulgatus"]

PIPELINE = """
**How the numbers are produced.** Each sample's qPCR genome copies (stored as log₁₀
copies mL⁻¹) are converted back to copies and then to *within-sample proportions*, so a
culture that merely grew denser is not counted as compositionally different. Every pair of
samples is then scored by **Bray–Curtis dissimilarity** (0 = identical make-up, 1 = no
species in common), giving a full table of pairwise differences that **PCoA** draws as a
map — dots close together are similar communities, and each axis is labelled with the share
of total variation it carries. **PERMANOVA** splits that total variation between Training,
Day and their interaction, reporting **R²** (the share each explains) and a P value obtained
by relabelling whole biological lineages; **PERMDISP** separately asks whether the groups
differ in *scatter* rather than position, which would otherwise masquerade as a
compositional difference.
"""


# ------------------------------------------------------------------ data helpers
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    return pd.read_csv(mv.DEFAULT_DATA)


def species_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in LABELS]


@st.cache_data(show_spinner="Recomputing…")
def run_analysis(df: pd.DataFrame, nperm: int, nboot: int):
    out = {}
    for model, g in df.groupby("community", sort=False):
        g = g.dropna(axis=1, how="all").reset_index(drop=True)
        g = g.rename(columns={"community": "model"})
        out[model] = mv.analyse(g, model, nperm=nperm, nboot=nboot)
    return out


# ----------------------------------------------------------------------- plots
def plot_ordination(results: dict):
    models = list(results)
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 4.4))
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        res = results[model]
        X, frac = res["coords"], res["pcoa_fraction"]
        lab = res["labels"].reset_index(drop=True)
        ax.axhline(0, color="#DDD", lw=0.8, zorder=0)
        ax.axvline(0, color="#DDD", lw=0.8, zorder=0)

        for training in ("Untrained", "Trained"):
            for day, marker in ((3, "o"), (6, "^")):
                m = ((lab.training == training) & (lab.day == day)).to_numpy()
                if not m.any():
                    continue
                c = PINK if training == "Trained" else GREY
                pts = X[m][:, :2]
                if len(pts) >= 3:
                    try:
                        h = ConvexHull(pts)
                        ax.add_patch(Polygon(pts[h.vertices], closed=True,
                                             facecolor=c, alpha=0.12,
                                             edgecolor=c, lw=0.8, zorder=1))
                    except Exception:
                        pass
                ax.scatter(pts[:, 0], pts[:, 1], marker=marker, s=70,
                           facecolor=c, edgecolor="#333", lw=0.7, zorder=3,
                           label=f"{training} Day {day}")

        txt = ["PERMANOVA"]
        for _, r in res["table"].iterrows():
            if r.term in ("Residual", "Total"):
                continue
            if r.term == "Day":
                continue
            txt.append(f"{r.term}: $R^2$={r.R2:.2f}, "
                       f"P={res['p_restricted'][r.term]:.2f}")
        ax.text(0.03, 0.03, "\n".join(txt), transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left")

        ax.set_xlabel(f"PCo1 ({frac[0] * 100:.1f}% of Bray–Curtis variation)",
                      fontsize=9)
        ax.set_ylabel(f"PCo2 ({frac[1] * 100:.1f}%)", fontsize=9)
        ax.set_title(mv.LABEL.get(model, model), fontsize=11, weight="bold")
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_raw(df: pd.DataFrame, model: str):
    g = df[df.community == model].dropna(axis=1, how="all")
    sp = [s for s in SPECIES_ORDER if s in species_cols(g)]
    days = sorted(g.day.unique())
    fig, axes = plt.subplots(1, len(sp), figsize=(2.1 * len(sp), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sp):
        for training, c in (("Untrained", GREY), ("Trained", PINK)):
            sub = g[g.training == training]
            for rep, r in sub.groupby("replicate"):
                r = r.sort_values("day")
                ax.plot(r.day, r[s], marker="o", ms=4, lw=1.0, color=c,
                        alpha=0.75)
            mean = sub.groupby("day")[s].mean()
            ax.plot(mean.index, mean.values, marker="s", ms=7, lw=2.2,
                    color=c, label=training)
        ax.set_title(s, fontsize=8, style="italic")
        ax.set_xticks(days)
        ax.set_xlabel("Day", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("log₁₀ copies mL⁻¹", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{mv.LABEL.get(model, model)} — raw abundances "
                 "(thin = lineages, thick = group mean)", fontsize=9)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- sensitivity
def amplify(df: pd.DataFrame, model: str, lam: float) -> pd.DataFrame:
    """Move each sample away from (lam>1) or toward (lam<1) its own Training-group
    mean, in log-abundance space, leaving group means unchanged.

    lam = 1 is the measured data. Lowering lam tightens replicates within a group,
    which is the only lever that can make a between-group effect look cleaner.
    """
    out = df.copy()
    m = out.community == model
    sp = [c for c in species_cols(out) if out.loc[m, c].notna().any()]
    for (_, _), idx in out[m].groupby(["training", "day"]).groups.items():
        block = out.loc[idx, sp]
        out.loc[idx, sp] = block.mean() + (block - block.mean()) * lam
    return out


# ------------------------------------------------------------------------- app
st.title("SIHUMI lactose training — multivariate community analysis")
st.markdown(PIPELINE)

if "data" not in st.session_state:
    st.session_state.data = load_default()

with st.sidebar:
    st.header("Settings")
    up = st.file_uploader("Load a CSV", type="csv")
    if up is not None and st.button("Use uploaded file"):
        st.session_state.data = pd.read_csv(up)
        st.rerun()
    nperm = st.select_slider("Permutations", [199, 999, 4999, 9999], value=999,
                             help="Affects only the Monte-Carlo P values. The "
                                  "lineage-constrained P is exact and never "
                                  "depends on this.")
    nboot = st.select_slider("Bootstrap resamples", [200, 1000, 5000], value=1000)
    st.caption("Lower values are faster; the published table used 9,999 / 5,000.")
    if st.button("Reset data to published values"):
        st.session_state.data = load_default()
        st.cache_data.clear()
        st.rerun()

df = st.session_state.data
edited = not df.equals(load_default())
if edited:
    st.warning("**Modified data.** The figures and tables below are computed from "
               "your edited values, not the measured dataset. Nothing here should "
               "be reported as experimental data.", icon="⚠️")

tabs = st.tabs(["Data", "Raw abundances", "Ordination", "Results",
                "Sensitivity"])

# ---- Data -----------------------------------------------------------------
with tabs[0]:
    st.subheader("Measured data")
    st.caption("log₁₀ genome copies mL⁻¹ at 24 h. Edit any cell — every plot and "
               "statistic recomputes from this table.")
    new = st.data_editor(df, use_container_width=True, num_rows="fixed",
                         disabled=LABELS, key="editor")
    if not new.equals(df):
        st.session_state.data = new
        st.rerun()
    st.download_button("Download this table as CSV",
                       df.to_csv(index=False).encode(),
                       "abundances.csv", "text/csv")

results = run_analysis(df, nperm, nboot)

# ---- Raw ------------------------------------------------------------------
with tabs[1]:
    st.subheader("Raw abundances by lineage")
    st.caption("Each thin line is one biological lineage followed from Day 3 to "
               "Day 6; the thick line is the group mean. Lineages that move "
               "together are the repeated-measures structure the PERMANOVA "
               "permutation scheme has to respect.")
    for model in results:
        if (df.community == model).sum() and df[df.community == model].day.nunique() > 1:
            st.pyplot(plot_raw(df, model))
        else:
            st.pyplot(plot_raw(df, model))

# ---- Ordination -----------------------------------------------------------
with tabs[2]:
    st.subheader("Principal coordinates analysis")
    st.pyplot(plot_ordination(results))
    st.caption("Grey = untrained, pink = trained; circles = Day 3, triangles = "
               "Day 6. Hulls aid visualisation only. Axis sign and orientation "
               "are arbitrary; only separation and overlap carry meaning, and "
               "panels are independent ordinations so coordinates are not "
               "comparable between them.")

# ---- Results --------------------------------------------------------------
with tabs[3]:
    st.subheader("Supplementary Table S4")
    s4 = mv.table_s4(list(results.values()))
    st.dataframe(s4, use_container_width=True, hide_index=True)

    st.markdown("""
**Reading the columns.** *R²* is the share of all community variation attributable to that
factor — Training R² = 0.76 means three-quarters of how these communities differ is
explained by whether they were trained. The *95% CI* comes from resampling whole lineages,
and is the honest statement of precision at n = 3. *Pseudo-F* is the between-group
separation divided by the within-group scatter. The *PERMANOVA P* is how often randomly
relabelled lineages do as well; with three lineages per arm only ten distinct relabellings
exist, so **0.100 is the smallest value this test can ever return for Training**, whatever
the effect size. *PERMDISP P* tests spread rather than position, reported against both the
spatial median (robust) and the centroid.
""")

    for model, res in results.items():
        with st.expander(f"{mv.LABEL.get(model, model)} — full output"):
            t = res["table"].copy()
            t["P (lineage)"] = [res["p_restricted"].get(x, np.nan) for x in t.term]
            t["P (unrestricted)"] = [res["p_free"].get(x, np.nan) for x in t.term]
            t["arrangements"] = [
                f"{res['arrangements'][x][1]}/{res['arrangements'][x][0]}"
                if x in res["arrangements"] else "" for x in t.term]
            st.markdown("**PERMANOVA**")
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.code(mv.interpret_permanova(res), language=None)
            st.markdown("**PERMDISP**")
            st.dataframe(pd.DataFrame([{
                "factor": d["factor"], "reference": d["kind"],
                "F": round(d["F"], 3), "P": d["p"],
                **{f"mean dist [{k}]": round(v, 4)
                   for k, v in d["mean_dist"].items()}}
                for d in res["permdisp"]]),
                use_container_width=True, hide_index=True)
            st.code(mv.interpret_permdisp(res), language=None)

# ---- Sensitivity ----------------------------------------------------------
with tabs[4]:
    st.subheader("How fragile is the result?")
    st.info("This tab is a **teaching sandbox**. It rescales the scatter between "
            "replicate lineages to show how the statistics respond, and never "
            "changes any group mean. The simulated values are not experimental "
            "data and are not exported.", icon="🧪")
    model = st.selectbox("Community model", list(results))
    lam = st.slider("Replicate scatter (×)", 0.1, 2.0, 1.0, 0.1,
                    help="1.0 = measured data. Below 1, replicates within each "
                         "group are tightened; above 1, they are spread out.")

    sim = amplify(df, model, lam)
    sub = sim[sim.community == model].dropna(axis=1, how="all") \
        .rename(columns={"community": "model"}).reset_index(drop=True)
    r = mv.analyse(sub, model, nperm=nperm, nboot=200)

    cols = st.columns(3)
    for col, (_, row) in zip(cols, r["table"].iterrows()):
        if row.term in ("Residual", "Total"):
            continue
        p = r["p_restricted"][row.term]
        base = results[model]["table"].set_index("term").loc[row.term, "R2"]
        col.metric(row.term, f"R² = {row.R2:.2f}", f"{row.R2 - base:+.2f} vs measured")
        col.caption(f"P = {p:.3f}")

    floor = 1.0 / r["arrangements"]["Training"][1]
    st.markdown(f"""
**The point this makes.** However tightly you squeeze the replicates, the
lineage-constrained P for Training cannot fall below **{floor:.3f}** — there are only
{r['arrangements']['Training'][1]} distinct ways to relabel six lineages into two groups of
three, so the most extreme possible outcome is 1 in {r['arrangements']['Training'][1]}. No
amount of cleaner data fixes that; only more independent lineages would. This is exactly
why the manuscript reports R² with confidence intervals as the primary evidence rather
than leaning on the P value.
""")
    st.pyplot(plot_ordination({model: r}))
