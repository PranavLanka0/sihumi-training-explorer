# SIHUMI lactose-training explorer

Interactive companion to the multivariate community analysis of a defined gut
microbial community trained with lactose: Bray–Curtis dissimilarity, principal
coordinates analysis (PCoA), PERMANOVA and PERMDISP.

It is a **teaching and exploration tool**. The authoritative analysis, the
dataset of record and the published tables live with the manuscript; see
*Relationship to the paper* below.

## What it does

| Tab | Contents |
|---|---|
| **Data** | The abundance table as an editable grid. Every plot and statistic recomputes from it. |
| **Raw abundances** | Per-species log₁₀ copies mL⁻¹, one thin line per biological lineage from Day 3 to Day 6, thick line = group mean. |
| **Ordination** | The three PCoA panels (SIHUMI, SIHUMI-4, monoculture-trained), annotated with PERMANOVA R² and P. |
| **Results** | Full PERMANOVA and PERMDISP output plus Table S4, each with a plain-language interpretation. |
| **Sensitivity** | Rescales *within-group* replicate scatter — group means are never moved — to show how the statistics respond, and why the permutation floor cannot be escaped. |

### The permutation floor

With three biological lineages per arm there are only
C(6,3) = 20 ways to relabel whole lineages between trained and untrained, ten of
them distinct. The smallest attainable P for the Training main effect is
therefore **1/10 = 0.100**, whatever the size of the effect — in these data
pseudo-F = 90.6 against a null concentrated at zero still returns P = 0.100.
This is a property of the design, not of the biology, and it is why effect sizes
(R²) with bootstrap confidence intervals are the primary evidence. The
Sensitivity tab demonstrates it directly.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The analysis engine also runs standalone, printing the full PERMANOVA and
PERMDISP tables with interpretations and writing five CSVs:

```bash
python NatiRios_multivariate_analysis.py --outdir results
```

With no arguments it additionally runs a self-check asserting that the published
sums of squares and R² values are recovered and that the R² terms sum to unity
within each model.

## Data

`data_relative_abundances.csv` — 30 samples, log₁₀ genome copies mL⁻¹ at 24 h.

| Column | Meaning |
|---|---|
| `community` | SIHUMI-7, SIHUMI-4, or Monoculture (monoculture-trained assembly) |
| `day` | 3 or 6 (either side of the lactose-free break) |
| `training` | Trained / Untrained |
| `replicate` | Biological lineage, 1–3 |
| *species columns* | log₁₀ genome copies mL⁻¹; empty where a species is not a member of that community |

Each replicate is a **serially propagated lineage**: replicate 1 at Day 3 and
replicate 1 at Day 6 are the same culture line. Trained and untrained lines are
always separate. The permutation scheme depends on this structure.

## Method

Abundances are converted to within-sample proportions, so a culture that merely
grew denser is not counted as compositionally different. Bray–Curtis
dissimilarities feed a classical (Torgerson) PCoA via eigendecomposition of the
Gower double-centred matrix. PERMANOVA uses a sequential (Type I) sum-of-squares
partition (McArdle & Anderson 2001); significance is assessed by permuting whole
lineages between training groups and day labels within lineages, by exact
enumeration (Anderson & ter Braak 2003). PERMDISP (Anderson 2006) is reported
against both the group spatial median — the default of `vegan::betadisper` — and
the group centroid. R² confidence intervals come from 5,000 bootstrap resamples
of whole lineages, stratified within training group.

Implemented directly in NumPy/SciPy/pandas, with no compiled ecology dependency,
so every step is open to inspection.

### References

- Anderson MJ (2001) *Austral Ecol* 26:32–46
- McArdle BH & Anderson MJ (2001) *Ecology* 82:290–297
- Anderson MJ & ter Braak CJF (2003) *J Stat Comput Simul* 73:85–113
- Anderson MJ (2006) *Biometrics* 62:245–253
- Kelly BJ et al. (2015) *Bioinformatics* 31:2461–2468

## Relationship to the paper

This repository is a companion, not the deposit of record. The dataset and the
analysis script archived with the manuscript (and cited in its Data and Code
Availability statement) are the authoritative versions.

Edits made in the Data tab, and anything produced by the Sensitivity tab, are
exploratory simulations for understanding the statistics. They are not
experimental data and must not be reported as such.
