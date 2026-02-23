# Climate Events and Firm Performance

This project studies how local climate events affect the financial and market performance of U.S. publicly listed firms, using a panel matched to historical disaster data (SHELDUS) and SEC 10-K filings.

A key benchmark is Hsu, Lee, Peng, and Yi (2018), "Natural Disasters, Technology Diversity, and Operating Performance," *Review of Economics and Statistics*, 100(4): 619–630.

## Research design

A firm is "treated" in the first year its headquarters county experiences a climate event above a severity threshold (e.g., 75th percentile of per-capita property damage). We estimate the causal effect on firm-level outcomes — log revenue, log assets, log market capitalization, and cash ratio — using three staggered difference-in-differences estimators:

1. **Sun and Abraham (2021)** interaction-weighted estimator (baseline)
2. **Stacked DiD** following Cengiz, Dube, Lindner, and Zipperer (2019) (robustness)
3. **de Chaisemartin and D'Haultfœuille (2024)** `did_multiplegt_dyn` (robustness)

## Project structure

```
├── README.md
├── research-proposal.md          # Full research proposal
├── data/                         # Processed panel datasets (Parquet)
│   ├── df_financial.parquet      # Compustat firm financials
│   ├── df_industry.parquet       # SIC industry codes
│   ├── df_location.parquet       # Firm HQ locations (from 10-K)
│   ├── df_sheldus.parquet        # SHELDUS climate events (county-level)
│   ├── example_treatment.parquet # Treatment assignments
│   ├── example_did_staggered.parquet  # Sun & Abraham panel
│   ├── example_did_stacked.parquet    # Cengiz et al. stacked panel
│   └── example_did_multi_event.parquet # dCDH multi-event panel
│
├── data-preparation/
│   ├── data_preparation.ipynb    # Full pipeline: load → match → build panels
│   └── climate_study/            # Helper modules
│       ├── treatment/            # SHELDUS treatment builder
│       └── visualization/        # Maps and financial plots
│
└── analysis/
    └── event_study/
        ├── 00_baseline_regressions.ipynb  # Main estimation notebook
        ├── r/
        │   ├── helpers.R                  # Estimation wrappers (sunab, roth, dCDH)
        │   └── dechaisemartin/            # Local copy of did_multiplegt_dyn R code
        └── diff_in_diff_package/          # Python dCDH package (vendored)
```

### `data-preparation/`

**Kernel:** `research_and_analytics` (Python 3.11)

Builds the analysis-ready datasets. The pipeline:
1. Loads Compustat financials, firm HQ locations (from 10-K), and SHELDUS county-level climate events.
2. Geocodes firm cities to county FIPS codes (Google Maps → FCC Census Block API).
3. Assigns treatment based on a severity threshold (configurable variable and percentile).
4. Constructs three panel formats: staggered (Sun & Abraham), stacked (Cengiz et al.), and multi-event (dCDH).

Panel construction uses [`did-panel-builder`](https://github.com/thomaspernet/did-panel-builder), a custom package for building DiD-ready panels from staggered treatment data.

### `analysis/event_study/`

**Kernel:** `diff_in_diff` (R 4.4, via conda)

Estimates treatment effects across three estimators. The main notebook is `00_baseline_regressions.ipynb`. Helper functions for estimation, formatting, and robustness are in `r/helpers.R`.

#### Note on `r/dechaisemartin/`

The R code in `analysis/event_study/r/dechaisemartin/` is a local copy of the R implementation from [Credible-Answers/did_multiplegt_dyn](https://github.com/Credible-Answers/did_multiplegt_dyn/tree/main/R/R). We vendor it locally because the upstream R package is not actively maintained and may not install cleanly. The code is used as-is without modifications.

## Environment setup

### 1. Python environment: `research_and_analytics`

Used for data preparation.

```bash
conda create -n research_and_analytics python=3.11 -y
conda activate research_and_analytics

pip install \
  pandas pyarrow geopandas \
  matplotlib seaborn \
  statsmodels scikit-learn \
  ipykernel \
  "did-panel-builder @ git+https://github.com/thomaspernet/did-panel-builder.git"
```

> [`did-panel-builder`](https://github.com/thomaspernet/did-panel-builder) is installed directly from GitHub.

Register the kernel for Jupyter/VS Code:

```bash
python -m ipykernel install --user --name research_and_analytics --display-name "research_and_analytics"
```

### 2. R environment: `diff_in_diff`

Used for estimation. R is installed via conda alongside the required R packages.

```bash
conda create -n diff_in_diff -c conda-forge \
  r-base=4.4 \
  r-tidyverse \
  r-fixest \
  r-arrow \
  r-marginaleffects \
  r-sandwich \
  r-cowplot \
  r-data.table \
  r-haven \
  r-matlib \
  r-plm \
  r-lmtest \
  r-irkernel \
  -y

conda activate diff_in_diff
```

Register the R kernel:

```bash
Rscript -e 'IRkernel::installspec(name = "diff_in_diff", displayname = "R: Diff-in-Diff")'
```

In VS Code, select the kernel **R: Diff-in-Diff** (located at `~/anaconda3/envs/diff_in_diff/lib/R/bin/R`) when opening the analysis notebooks.

## Replication

1. Clone the repository.
2. Create both conda environments (see above).
3. Open `data-preparation/data_preparation.ipynb` with the `research_and_analytics` kernel and run all cells to produce the panel datasets in `data/`.
4. Open `analysis/event_study/00_baseline_regressions.ipynb` with the `R: Diff-in-Diff` kernel and run all cells.

## References

- Cengiz, D., Dube, A., Lindner, A., and Zipperer, B. (2019). "The Effect of Minimum Wages on Low-Wage Jobs." *Quarterly Journal of Economics*, 134(3): 1405–1454.
- de Chaisemartin, C., and D'Haultfœuille, X. (2024). "Difference-in-Differences Estimators of Intertemporal Treatment Effects." *Review of Economics and Statistics*, forthcoming.
- Hsu, P.-H., Lee, H.-H., Peng, S.-C., and Yi, L. (2018). "Natural Disasters, Technology Diversity, and Operating Performance." *Review of Economics and Statistics*, 100(4): 619–630.
- Sun, L., and Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2): 175–199.
