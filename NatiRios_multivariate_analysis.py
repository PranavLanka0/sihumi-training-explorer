"""
Multivariate community analysis for ISMEJ-D-26-01764
"Training a defined gut microbial community with lactose ..."

Reproduces Supplementary Table S4 and the ordination in Figure 6 from the
strain-resolved qPCR abundances.

Pipeline
    log10 genome copies/mL  ->  copies/mL  ->  within-sample proportions
    -> Bray-Curtis dissimilarity
    -> classical (Torgerson) PCoA
    -> factorial PERMANOVA (sequential Type I SS; McArdle & Anderson 2001)
         * unrestricted permutation of samples
         * lineage-restricted permutation (exact enumeration)
    -> PERMDISP (Anderson 2006), group centroid and group spatial (L1) median
    -> bootstrap 95% CI for R^2 (resampling whole lineages)

Design
    3 biological replicates per arm. Each replicate is a serially propagated
    LINEAGE sampled on Day 3 and Day 6; Trained and Untrained are always
    separate lineages. Lineage id = (training, replicate).

Environment
    Python 3.12.4, numpy 1.26.4, scipy 1.13.1, pandas 2.2.2

Usage
    python isme_multivariate_analysis.py [path/to/PCoA.xlsx] [--outdir DIR]
Runs a self-check against the published values when called with no arguments.

References
    Anderson MJ (2001) Austral Ecol 26:32-46
    McArdle BH & Anderson MJ (2001) Ecology 82:290-297
    Anderson MJ & ter Braak CJF (2003) J Stat Comput Simul 73:85-113
    Anderson MJ (2006) Biometrics 62:245-253
    Anderson MJ (2017) Wiley StatsRef, doi:10.1002/9781118445112.stat07841
    Kelly BJ et al. (2015) Bioinformatics 31:2461-2468
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

SEED = 20260910
NPERM = 9999
NBOOT = 5000

# The dataset ships alongside this script in the repository.
DEFAULT_DATA = Path(__file__).resolve().parent / "data_relative_abundances.csv"

LABELS = ("community", "day", "training", "replicate")


# --------------------------------------------------------------------------- data
def load_data(path: Path) -> dict[str, pd.DataFrame]:
    """Load the dataset, keyed by community model.

    Accepts the tidy CSV that ships with this script, or the original
    block-structured workbook.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return load_blocks(path)

    df = pd.read_csv(path)
    df = df.rename(columns={"community": "model"})
    out = {}
    for model, g in df.groupby("model", sort=False):
        # species absent from a community model are empty in the shared CSV
        out[model] = g.dropna(axis=1, how="all").reset_index(drop=True)
    return out


def load_blocks(xlsx: Path) -> dict[str, pd.DataFrame]:
    """Parse the block-structured workbook into tidy tables, one per community model.

    Each block is: a title row ('<model>', 'Day N Untrained', ..., 'Day N Trained'),
    a species header row, then one row per biological replicate. Values are
    log10 genome copies/mL.
    """
    raw = pd.read_excel(xlsx, header=None)
    out: dict[str, list[dict]] = {}

    r = 0
    while r < len(raw):
        title = raw.iat[r, 0]
        if not isinstance(title, str) or not title.strip():
            r += 1
            continue

        model = title.strip()
        banner = [str(v) if isinstance(v, str) else "" for v in raw.iloc[r].tolist()]
        header = [str(v) if isinstance(v, str) else "" for v in raw.iloc[r + 1].tolist()]

        # column index where each "Day N <Training>" sub-block starts
        starts = [(j, banner[j]) for j in range(1, len(banner)) if banner[j].strip()]
        species = [h for h in header[1:] if h.strip()]
        n_sp = len(species) // len(starts)

        for j, label in starts:
            day = int(label.split()[1])
            training = label.split()[-1]          # 'Trained' / 'Untrained'
            cols = list(range(j, j + n_sp))
            names = [header[c] for c in cols]
            for rr in range(r + 2, len(raw)):
                rep = raw.iat[rr, 0]
                if pd.isna(rep) or not str(rep).strip().isdigit():
                    break
                vals = raw.iloc[rr, cols].astype(float).to_numpy()
                rec = {"model": model, "day": day, "training": training,
                       "replicate": int(rep)}
                rec.update(dict(zip(names, vals)))
                out.setdefault(model, []).append(rec)
        r += 2

    return {m: pd.DataFrame(v) for m, v in out.items()}


def proportions(df: pd.DataFrame) -> np.ndarray:
    """log10 copies/mL -> copies/mL -> within-sample proportions (rows sum to 1)."""
    sp = [c for c in df.columns if c not in ("model", "day", "training", "replicate")]
    counts = np.power(10.0, df[sp].to_numpy(float))
    return counts / counts.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ core matrices
def bray_curtis(P: np.ndarray) -> np.ndarray:
    return squareform(pdist(P, metric="braycurtis"))


def gower(D: np.ndarray) -> np.ndarray:
    """G = -0.5 * J D^2 J, the double-centred matrix PERMANOVA partitions."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return J @ (-0.5 * D ** 2) @ J


def pcoa(D: np.ndarray):
    """Classical (Torgerson) PCoA. Returns coordinates on positive axes and
    the fraction of variation each carries."""
    vals, vecs = np.linalg.eigh(gower(D))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = vals > 1e-10
    coords = vecs[:, pos] * np.sqrt(vals[pos])
    return coords, vals[pos] / vals[pos].sum()


# ------------------------------------------------------------------- PERMANOVA
def _contrasts(training: np.ndarray, day: np.ndarray | None, factorial: bool):
    """+-1 contrast columns. The design is balanced, so these are orthogonal and
    the sequential (Type I) partition does not depend on term order."""
    t = np.where(training == "Trained", 1.0, -1.0)
    cols = {"Training": t}
    if factorial:
        d = np.where(day == 3, 1.0, -1.0)
        cols["Day"] = d
        cols["Training x Day"] = t * d
    return cols


def permanova(G: np.ndarray, training, day, factorial: bool):
    """Sequential SS partition of the Gower matrix.

    For an orthogonal contrast column x, the projection is H = xx'/(x'x) and
    SS = tr(H G H) = x'Gx / (x'x).
    """
    cols = _contrasts(training, day, factorial)
    n = G.shape[0]
    total = np.trace(G)
    ss = {k: float(x @ G @ x / (x @ x)) for k, x in cols.items()}
    df_res = n - 1 - len(cols)
    ss_res = total - sum(ss.values())

    rows = []
    for k, v in ss.items():
        rows.append({"term": k, "df": 1, "SS": v, "R2": v / total,
                     "pseudo_F": (v / 1) / (ss_res / df_res)})
    rows.append({"term": "Residual", "df": df_res, "SS": ss_res,
                 "R2": ss_res / total, "pseudo_F": np.nan})
    rows.append({"term": "Total", "df": n - 1, "SS": total, "R2": 1.0,
                 "pseudo_F": np.nan})
    return pd.DataFrame(rows)


def _f_terms(G, training, day, factorial) -> dict[str, float]:
    cols = _contrasts(training, day, factorial)
    total = np.trace(G)
    ss = {k: float(x @ G @ x / (x @ x)) for k, x in cols.items()}
    df_res = G.shape[0] - 1 - len(cols)
    ss_res = total - sum(ss.values())
    return {k: (v / (ss_res / df_res) if ss_res > 1e-12 else np.inf)
            for k, v in ss.items()}


# ------------------------------------------------------------- permutation tests
def p_unrestricted(G, training, day, factorial, nperm=NPERM, seed=SEED):
    """All samples exchangeable (the scheme used in the original submission)."""
    rng = np.random.default_rng(seed)
    obs = _f_terms(G, training, day, factorial)
    n = len(training)
    ge = {k: 1 for k in obs}
    seen = {k: set() for k in obs}
    for _ in range(nperm):
        idx = rng.permutation(n)
        f = _f_terms(G, training[idx], (day[idx] if factorial else None), factorial)
        for k in obs:
            seen[k].add(round(f[k], 9))
            if f[k] >= obs[k] - 1e-12:
                ge[k] += 1
    return ({k: ge[k] / (nperm + 1) for k in obs},
            {k: len(seen[k]) for k in obs})


def p_lineage_restricted(G, training, day, lineage, factorial):
    """Split-plot scheme (Anderson & ter Braak 2003), by exact enumeration.

    Training is a WHOLE-PLOT factor: permute entire lineages between Training
    levels.  Day and Training x Day are SPLIT-PLOT: permute the Day labels
    within each lineage independently.

    At n = 3 per arm there are C(6,3) = 20 whole-lineage relabellings, 10 of them
    distinct by symmetry, so the smallest attainable P for Training is 1/10 = 0.10
    regardless of effect size.
    """
    obs = _f_terms(G, training, day, factorial)
    lin = np.asarray(lineage)
    uniq = sorted(set(lin))
    out, counts = {}, {}

    # --- Training: relabel whole lineages -----------------------------------
    n_tr = sum(training[lin == L][0] == "Trained" for L in uniq)
    stats = []
    for combo in itertools.combinations(range(len(uniq)), n_tr):
        lab = np.array(["Untrained"] * len(lin), dtype=object)
        for i in combo:
            lab[lin == uniq[i]] = "Trained"
        stats.append(_f_terms(G, lab, day, factorial)["Training"])
    stats = np.array(stats)
    out["Training"] = float((stats >= obs["Training"] - 1e-12).sum() / len(stats))
    counts["Training"] = (len(stats), len(set(np.round(stats, 9))))

    # --- Day and interaction: flip Day within each lineage ------------------
    if factorial:
        for term in ("Day", "Training x Day"):
            stats = []
            for flips in itertools.product([False, True], repeat=len(uniq)):
                d = day.copy()
                for L, flip in zip(uniq, flips):
                    if flip:
                        m = lin == L
                        d[m] = np.where(d[m] == 3, 6, 3)
                stats.append(_f_terms(G, training, d, factorial)[term])
            stats = np.array(stats)
            out[term] = float((stats >= obs[term] - 1e-12).sum() / len(stats))
            counts[term] = (len(stats), len(set(np.round(stats, 9))))
    return out, counts


# ---------------------------------------------------------------------- PERMDISP
def _spatial_median(X, iters=256, tol=1e-10):
    """Weiszfeld's algorithm: the point minimising total distance to all rows.
    This is what vegan::betadisper uses by default."""
    m = X.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(X - m, axis=1)
        d = np.where(d < 1e-12, 1e-12, d)
        new = (X / d[:, None]).sum(axis=0) / (1.0 / d).sum()
        if np.linalg.norm(new - m) < tol:
            return new
        m = new
    return m


def permdisp(D: np.ndarray, groups, kind="centroid", nperm=NPERM, seed=SEED):
    """Anderson (2006). Distances to the group centre in PCoA space, then a
    one-way F on those distances with a permutation p-value."""
    X, _ = pcoa(D)
    groups = np.asarray(groups)
    dist = np.empty(len(groups))
    for g in set(groups):
        m = groups == g
        centre = X[m].mean(axis=0) if kind == "centroid" else _spatial_median(X[m])
        dist[m] = np.linalg.norm(X[m] - centre, axis=1)

    def F(d, lab):
        levels = sorted(set(lab))
        grand = d.mean()
        between = sum((lab == g).sum() * (d[lab == g].mean() - grand) ** 2 for g in levels)
        within = sum(((d[lab == g] - d[lab == g].mean()) ** 2).sum() for g in levels)
        df_b, df_w = len(levels) - 1, len(d) - len(levels)
        return (between / df_b) / (within / df_w) if within > 1e-12 else np.inf

    obs = F(dist, groups)
    rng = np.random.default_rng(seed)
    ge = 1 + sum(F(dist, groups[rng.permutation(len(groups))]) >= obs - 1e-12
                 for _ in range(nperm))
    means = {str(g): float(dist[groups == g].mean()) for g in sorted(set(groups))}
    return {"F": float(obs), "p": ge / (nperm + 1), "kind": kind, "mean_dist": means}


# --------------------------------------------------------------------- bootstrap
def bootstrap_r2(P, training, day, lineage, factorial, nboot=NBOOT, seed=SEED):
    """Cluster bootstrap: resample whole LINEAGES with replacement, so each
    lineage's Day 3 / Day 6 pair always travels together and the repeated-measures
    structure is preserved. Percentile 95% CI.

    Resamples in which the residual SS collapses to zero (possible when the same
    lineage is drawn repeatedly) are undefined and are dropped; the number of
    valid resamples is reported.
    """
    rng = np.random.default_rng(seed)
    lin = np.asarray(lineage)
    # Stratified by Training arm: draw lineages with replacement WITHIN each arm, so
    # every resample keeps the balanced design (3 trained + 3 untrained lineages).
    # Pooling the six lineages instead would allow resamples with an arm missing, for
    # which R^2 is undefined.
    arms = {a: sorted(set(lin[training == a])) for a in sorted(set(training))}
    keep: dict[str, list[float]] = {}

    for _ in range(nboot):
        picked = [u[i] for u in arms.values()
                  for i in rng.integers(0, len(u), len(u))]
        idx = np.concatenate([np.flatnonzero(lin == L) for L in picked])
        G = gower(bray_curtis(P[idx]))
        if np.trace(G) < 1e-12:
            continue
        tab = permanova(G, training[idx], day[idx] if factorial else None, factorial)
        res = tab.loc[tab.term == "Residual", "SS"].iloc[0]
        if res < 1e-12:
            continue                      # undefined pseudo-F; drop
        for _, row in tab.iterrows():
            if row.term not in ("Residual", "Total"):
                keep.setdefault(row.term, []).append(row.R2)

    return {k: {"CI95_low": float(np.percentile(v, 2.5)),
                "CI95_high": float(np.percentile(v, 97.5)),
                "median_boot": float(np.median(v)),
                "n_valid_boot": len(v)} for k, v in keep.items()}


# -------------------------------------------------------------------------- main
def analyse(df: pd.DataFrame, model: str, nperm=NPERM, nboot=NBOOT):
    factorial = df.day.nunique() > 1
    P = proportions(df)
    D = bray_curtis(P)
    G = gower(D)
    training = df.training.to_numpy()
    day = df.day.to_numpy()
    lineage = np.array([f"{t}_{r}" for t, r in zip(training, df.replicate)])

    table = permanova(G, training, day if factorial else None, factorial)
    p_free, uniq_free = p_unrestricted(G, training, day, factorial, nperm=nperm)
    p_rest, counts = p_lineage_restricted(G, training, day, lineage, factorial)
    boot = bootstrap_r2(P, training, day, lineage, factorial, nboot=nboot)

    disp = []
    for factor, lab in ([("Training", training)] +
                        ([("Day", day)] if factorial else [])):
        for kind in ("centroid", "median"):
            r = permdisp(D, lab, kind=kind, nperm=nperm)
            disp.append({"model": model, "factor": factor, **r})

    coords, frac = pcoa(D)
    return {"model": model, "table": table, "p_free": p_free,
            "unique_free": uniq_free, "p_restricted": p_rest,
            "arrangements": counts, "bootstrap": boot, "permdisp": disp,
            "pcoa_fraction": frac[:3].tolist(), "coords": coords,
            "bray_curtis": D, "labels": df[list(LABELS[1:])].copy()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=str(DEFAULT_DATA),
                    help="tidy CSV (default) or the original .xlsx")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)

    blocks = load_data(Path(args.data))
    name_map = {"SIHUMI-7": "SIHUMI-7", "SIHUMI-4": "SIHUMI-4",
                "Monoculture": "Assembly"}

    results, rows_tab, rows_scn, rows_disp, rows_boot = [], [], [], [], []
    for raw_name, df in blocks.items():
        model = name_map.get(raw_name, raw_name)
        res = analyse(df.reset_index(drop=True), model)
        results.append(res)

        for _, r in res["table"].iterrows():
            rows_tab.append({"model": model, **r.to_dict()})
        for term, p in res["p_free"].items():
            n_tot, n_uni = res["arrangements"][term]
            rows_scn.append({
                "model": model, "term": term,
                "p_unrestricted": p, "unique_perms_unrestricted": res["unique_free"][term],
                "p_lineage_restricted": res["p_restricted"][term],
                "arrangements": n_tot, "arrangements_unique": n_uni})
        for d in res["permdisp"]:
            rows_disp.append({k: (json.dumps(v) if isinstance(v, dict) else v)
                              for k, v in d.items()})
        for term, b in res["bootstrap"].items():
            rows_boot.append({"model": model, "term": term, **b})

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_tab).to_csv(out / "01_PERMANOVA_table.csv", index=False)
    pd.DataFrame(rows_scn).to_csv(out / "02_permutation_scenarios.csv", index=False)
    pd.DataFrame(rows_disp).to_csv(out / "03_PERMDISP.csv", index=False)
    pd.DataFrame(rows_boot).to_csv(out / "04_bootstrap_R2_CI.csv", index=False)

    s4 = table_s4(results)
    s4.to_csv(out / "05_Table_S4.csv", index=False)
    report(results, s4)
    print(f"\nCSV output written to {out.resolve()}")
    return results


# ------------------------------------------------------------------- reporting
LABEL = {"SIHUMI-7": "SIHUMI (7 species)", "SIHUMI-4": "SIHUMI-4 (4 species)",
         "Assembly": "SIHUMI-4 (monoculture training)"}
def table_s4(results) -> pd.DataFrame:
    """Supplementary Table S4 as it appears in the manuscript.

    PERMANOVA P is the lineage-constrained value. PERMDISP is reported against
    both references: the group spatial median (the default of betadisper in the
    R package vegan) and the group centroid.
    """
    rows = []
    for res in results:
        disp = {(d["factor"], d["kind"]): d for d in res["permdisp"]}
        for _, r in res["table"].iterrows():
            if r.term in ("Residual", "Total"):
                continue
            b = res["bootstrap"].get(r.term, {})
            med = disp.get((r.term, "median"))
            cen = disp.get((r.term, "centroid"))
            rows.append({
                "Community Model": LABEL.get(res["model"], res["model"]),
                "Design & Factor": r.term,
                "R2 (Variance Explained)": f"{r.R2:.2f}",
                "R2 95% CI": (f"{b['CI95_low']:.2f}-{b['CI95_high']:.2f}"
                              if b else "-"),
                "Pseudo F": f"{r.pseudo_F:.1f}",
                "PERMANOVA P value": f"{res['p_restricted'][r.term]:.3f}",
                "PERMDISP P (spatial median)": (f"{med['p']:.3f}" if med else "-"),
                "PERMDISP P (centroid)": (f"{cen['p']:.3f}" if cen else "-"),
            })
    return pd.DataFrame(rows)


def _size(r2):
    return ("negligible" if r2 < 0.05 else "small" if r2 < 0.15 else
            "moderate" if r2 < 0.35 else "large" if r2 < 0.65 else "very large")


def interpret_permanova(res) -> str:
    """Plain-language reading of each PERMANOVA term."""
    lines = ["\n  Interpretation"]
    for _, r in res["table"].iterrows():
        if r.term in ("Residual", "Total"):
            continue
        b = res["bootstrap"].get(r.term, {})
        p = res["p_restricted"][r.term]
        n_uni = res["arrangements"][r.term][1]
        floor = 1.0 / n_uni
        ci = (f"95% CI {b['CI95_low']:.2f}-{b['CI95_high']:.2f}" if b else "")
        verdict = ("at the permutation floor, so this P is the smallest value the "
                   "test can return and carries no evidence either way"
                   if abs(p - floor) < 1e-9 else
                   "below the conventional 0.05 threshold" if p < 0.05 else
                   "above the conventional 0.05 threshold")
        lines.append(
            f"    {r.term:<16s} explains {r.R2 * 100:4.1f}% of the Bray-Curtis "
            f"variation ({_size(r.R2)} effect; {ci}).")
        lines.append(
            f"    {'':<16s} P = {p:.3f} from {n_uni} distinct lineage arrangements "
            f"(floor {floor:.3f}) - {verdict}.")
    lines.append("    Residual is the variation among replicate lineages within a "
                 "Training x Day cell: the noise the effects are judged against.")
    return "\n".join(lines)


def interpret_permdisp(res) -> str:
    """Plain-language reading of each dispersion test."""
    lines = ["\n  Interpretation"]
    for factor in dict.fromkeys(d["factor"] for d in res["permdisp"]):
        got = {d["kind"]: d for d in res["permdisp"] if d["factor"] == factor}
        med, cen = got["median"], got["centroid"]
        spread = ", ".join(f"{k} {v:.3f}" for k, v in med["mean_dist"].items())
        widest = max(med["mean_dist"], key=med["mean_dist"].get)
        agree = (med["p"] < 0.05) == (cen["p"] < 0.05)
        lines.append(
            f"    {factor:<10s} mean distance to the group centre: {spread} "
            f"-> '{widest}' is the more variable group.")
        lines.append(
            f"    {'':<10s} P = {med['p']:.3f} (spatial median), "
            f"{cen['p']:.3f} (centroid)"
            + ("." if agree else
               " - the two disagree, which happens when one replicate sits far from "
               "the rest: the median is robust to it, the centroid is not."))
        if med["p"] < 0.05:
            lines.append(
                f"    {'':<10s} Groups differ in SPREAD as well as position, so a "
                f"PERMANOVA effect on {factor} is partly a consistency difference.")
    return "\n".join(lines)


def report(results, s4):
    """Print the full PERMANOVA and PERMDISP results, then Table S4."""
    for res in results:
        t = res["table"].copy()
        t["P_unrestricted"] = [res["p_free"].get(x, np.nan) for x in t.term]
        t["P_lineage"] = [res["p_restricted"].get(x, np.nan) for x in t.term]
        t["arrangements"] = [
            f"{res['arrangements'][x][1]}/{res['arrangements'][x][0]}"
            if x in res["arrangements"] else "" for x in t.term]
        t["CI95_low"] = [res["bootstrap"].get(x, {}).get("CI95_low", np.nan)
                         for x in t.term]
        t["CI95_high"] = [res["bootstrap"].get(x, {}).get("CI95_high", np.nan)
                          for x in t.term]

        print(f"\n{'=' * 78}\n  {LABEL.get(res['model'], res['model'])}"
              f"   (PCoA axes: " +
              ", ".join(f"{f * 100:.1f}%" for f in res["pcoa_fraction"]) + ")"
              f"\n{'=' * 78}")
        print("\n  PERMANOVA  (sequential Type I SS on the Gower-centred matrix)")
        print(t.to_string(index=False, na_rep="",
                          float_format=lambda v: f"{v:.4f}"))
        print(f"  sum of R^2 (excl. Total) = "
              f"{t[t.term != 'Total'].R2.sum():.6f}   (must be 1.000000)")
        print("  arrangements = unique/total distinct label assignments under "
              "the lineage-constrained scheme")

        print(interpret_permanova(res))

        print("\n  PERMDISP  (Anderson 2006; 9,999 permutations)")
        d = pd.DataFrame([{
            "factor": x["factor"], "reference": x["kind"], "F": x["F"],
            "P": x["p"],
            **{f"mean dist [{k}]": v for k, v in x["mean_dist"].items()},
        } for x in res["permdisp"]])
        print(d.to_string(index=False, na_rep="",
                          float_format=lambda v: f"{v:.4f}"))
        print(interpret_permdisp(res))

    print(f"\n{'=' * 78}\n  Supplementary Table S4  "
          f"(PERMDISP: both references)\n{'=' * 78}")
    print(s4.to_string(index=False))


def self_check(results):
    """Assert the published values are reproduced. Deterministic quantities only;
    permutation p-values are checked loosely because they are Monte Carlo."""
    want = {  # model -> term -> (SS, R2)
        "SIHUMI-7": {"Training": (0.20676, 0.3559), "Day": (0.10987, 0.1891),
                     "Training x Day": (0.01614, 0.0278)},
        "SIHUMI-4": {"Training": (0.14320, 0.7649), "Day": (0.01579, 0.0844),
                     "Training x Day": (0.01557, 0.0832)},
        "Assembly": {"Training": (0.03261, 0.5043)},
    }
    for res in results:
        t = res["table"].set_index("term")
        assert abs(t.drop("Total").R2.sum() - 1.0) < 1e-9, \
            f"{res['model']}: R^2 must sum to 1"
        for term, (ss, r2) in want[res["model"]].items():
            assert abs(t.loc[term, "SS"] - ss) < 5e-5, (res["model"], term, "SS")
            assert abs(t.loc[term, "R2"] - r2) < 5e-4, (res["model"], term, "R2")
        # the permutation floor: Training can never go below 1/10 at n = 3
        assert res["p_restricted"]["Training"] >= 0.0999, res["model"]
    print("\nself-check OK: Supplementary Table S4 values reproduced, "
          "all R^2 sum to 1, permutation floor holds.")


if __name__ == "__main__":
    res = main()
    if len(sys.argv) == 1:
        self_check(res)
