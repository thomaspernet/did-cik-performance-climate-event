# Climate Events and Firm Performance – Research Proposal (Master 1)

## 1. Motivation and Benchmark Paper
This project studies how local climate events affect firms’ financial and market performance, using a panel of U.S. firms matched to historical disaster data and corporate reports. A key benchmark is:

- Hsu, Lee, Peng, and Yi (2018), “Natural Disasters, Technology Diversity, and Operating Performance,” *Review of Economics and Statistics*, 100(4): 619–630.  
  The paper analyzes the effect of U.S. natural disasters on firms’ operating performance and shows that firms with more diversified technologies are more resilient.

## 2. Research Question and Contribution

### Main research question
Do climate events that affect a firm’s locations (e.g., headquarters or main operational sites) have a measurable impact on its financial and market performance, and how persistent are these effects over time?

### Possible extensions
- **Heterogeneity by sector:** manufacturing vs. services; energy-intensive vs. low-carbon sectors.
- **Heterogeneity by firm characteristics:** size, leverage, cash holdings, ESG profile (if available).
- **Outcome domain differences:** market-based outcomes (stock returns, volatility) vs. accounting outcomes (profitability, investment).

### Contribution
This project contributes by combining detailed climate event data with firm-level financials and 10-K qualitative information, and by implementing modern staggered-treatment difference-in-differences estimators (Sun & Abraham; de Chaisemartin & D’Haultfoeuille).

## 3. Data Description

### 3.1 Firm-level financials: Compustat
- **Coverage:** U.S. publicly listed firms over multiple decades.
- **Frequency:** annual firm-level panel.
- **Key variables (examples among the ≈500 available):**
  - **Income statement:** sales, operating income, net income.
  - **Balance sheet:** total assets, PPE, cash, leverage.
  - **Market variables:** market capitalization, book-to-market, returns (possibly merged from CRSP).
- **Main outcome variables:**
  - **Accounting performance:** ROA, profit margin, asset growth, investment (CAPEX).
  - **Market performance:** buy-and-hold abnormal returns around events; longer-run cumulative returns; possibly volatility.

### 3.2 Climate events: SHELDUS
- **Source:** Spatial Hazard Events and Losses Database for the United States (SHELDUS), 1960–2020.
- **Content:** county-level records of hazard events such as hurricanes, floods, wildfires, tornadoes, droughts, and associated damages.
- **Key dimensions:** event type, location (county, state), date, and damage measures (property losses, crop losses, fatalities).
- **Main variables for the project:**
  - Indicator that a city (or state) is affected by at least one event in a given year.
  - Intensity measures, e.g., total dollar losses in a city (or state)-year, or number of events.

#### Construction of treatment
A firm is “treated” in a given year if the city (or set of cities or states) associated with its location(s) experiences a climate event above a chosen intensity threshold.

### 3.3 10-K reports and firm locations
- **Source:** SEC 10-K filings, 2005–2020.
- **Available information:** headquarters location (and potentially later, operational locations such as plants, offices, data centers).
- **For this project version:** only headquarters location is used for firm-year geolocation.

#### Possible use of text data (optional extension)
- Extract climate- or risk-related terms from 10-K narratives to construct measures of climate risk disclosure or risk salience.
- Interact climate events with disclosure intensity to test whether more transparent firms show different market reactions.

## 4. Empirical Design

### 4.1 Treatment definition and sample construction
- Match each firm-year in Compustat to a city (or state) using the headquarters location from its 10-K in the corresponding year.
- Merge the firm–city (or state) panel with SHELDUS by city (or state)-year (and possibly event type).
- Define treatment indicators, for example:
  - **Any event:** indicator equal to 1 if the firm’s city (or state) experiences at least one climate event in year *t*.
  - **Intense event:** indicator equal to 1 if event losses in the city (or state) exceed a given percentile of the historical loss distribution.
- Construct an unbalanced panel of firms from 1960–2020 (or a shorter subperiod if dictated by 10-K coverage), with annual outcomes and treatment status.

### 4.2 Baseline outcomes and controls
**Baseline outcomes**
- ROA, operating margin, sales growth, CAPEX/Assets.
- Annual stock returns (if merged with CRSP) or cumulative returns over horizons after the event.

**Controls (depending on data availability)**
- Firm size (log assets), leverage, cash holdings.
- Sector fixed effects (e.g., 2-digit SIC), firm fixed effects, year fixed effects.

## 5. Econometric Strategy
Climate events generate staggered (and possibly repeated) treatments: different firms are treated in different years, and some may experience multiple events. Traditional two-way fixed effects (TWFE) DiD estimators can be biased under heterogeneous treatment effects. The project therefore uses modern estimators designed for staggered adoption and heterogeneous effects.

### 5.1 Event-study specification and Sun & Abraham estimator
**Objective:** estimate dynamic effects of climate events on firm outcomes before and after the event.

**Setup**
- Define event time:  
  \[
  k = t - T_i
  \]
  where \(T_i\) is the first year firm \(i\) is affected by a climate event (or first intense event).
- Estimate coefficients for leads and lags relative to \(T_i\), e.g.:
  \[
  k \in \{-3,-2,-1,0,1,2,3,\dots\}
  \]

**Estimator:** Sun & Abraham (2021) event-study estimator for staggered adoption.
- Corrects bias of standard TWFE event-study when effects vary across cohorts and over time.
- Implemented by interacting cohort-specific event-time dummies and aggregating appropriately.

**Interpretation**
- Pre-event coefficients test for parallel trends (placebo effects).
- Post-event coefficients trace the dynamic response of performance and market variables.

### 5.2 De Chaisemartin & D’Haultfoeuille estimator
**Purpose:** provide alternative and robust average treatment effect estimates in staggered or multi-period DiD.

**Design**
- Constructs weighted averages of valid 2×2 DiD comparisons at each time period.
- Avoids negative weights that can arise in TWFE.
- Can handle treatment intensity variation and, depending on implementation, repeated treatments or treatment reversals.

**Use in the project**
- Estimate average contemporaneous and cumulative effects of climate events on firm performance.
- Compare with Sun & Abraham event-study results as a robustness check.

## 6. Main Hypotheses
- **H1:** Climate events that directly affect a firm’s location reduce short-run operating performance (e.g., ROA, sales growth) and possibly reduce investment.
- **H2:** Market performance (stock returns) reacts negatively around severe events, but magnitude and persistence depend on firm characteristics.
- **H3 (optional extensions):**
  - Firms in more climate-exposed sectors, with higher leverage or lower liquidity, suffer stronger and more persistent effects.
  - Firms with stronger climate-related disclosures in 10-Ks exhibit different market reactions (e.g., smaller surprise component).

## 7. Feasibility and Expected Outputs

### Data and feasibility
- The project uses widely used datasets (Compustat, SHELDUS, SEC 10-K filings), making the design credible and replicable.
- The benchmark paper by Hsu et al. demonstrates that linking disaster data to firm-level financials is feasible in a high-quality journal setting, supporting feasibility for a Master-level project.

### Expected outputs
- Descriptive statistics and maps of climate events by region and time, and of treated vs. untreated firms.
- Event-study graphs showing dynamic effects of climate events on firm-level outcomes using the Sun & Abraham estimator.
- Average treatment effect estimates using de Chaisemartin & D’Haultfoeuille, with robustness checks across treatment definitions and subsamples.