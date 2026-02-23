# ==============================================================================
# helpers.R — Estimation wrappers, formatting, and robustness functions
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(fixest)
  library(arrow)
  library(ggplot2)
})

# ------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------

prepare_sunab <- function(df_raw) {

  df <- df_raw %>%
    mutate(
      cik          = as.factor(cik),
      sic          = as.factor(sic2),
      year         = as.numeric(year),
      year_treated = as.numeric(first_event_time),
      event_time   = ifelse(year_treated == -1000, NA, year - year_treated),
      group_id     = as.integer(factor(cik))
    ) %>%
    mutate(
      never_treated = as.numeric(year_treated == -1000),
      treated       = as.numeric(year_treated != -1000),
      tt_true       = year - year_treated
    )

#  df_sun <- df %>%
#    filter(treatment_type %in% c("never treated", "treated")) %>%
#    filter(is.na(event_time) | (event_time >= min_year & event_time <= max_year))

#  df_sun_filter <- filter_pre_post(df_sun,
#                                   min_pre_years  = min_pre_years,
#                                   min_post_years = min_post_years)

  list(
    df_full   = df
    #df_sun    = df_sun,
    #df_filter = df_sun_filter
  )
}

prepare_roth <- function(df_raw) {

  df <- df_raw %>%
    mutate_if(is.character, as.factor) %>%
    mutate_at(vars(starts_with("fe")), as.factor) %>%
    mutate(
      cik          = as.factor(cik),
      sic          = as.factor(sic2),
      year         = as.numeric(year),
      #year_treated = as.numeric(year_treated),
      group_id     = as.integer(factor(cik)),
      id_group     = interaction(cik, cohort, drop = TRUE),
      group_year   = interaction(cohort, year, drop = TRUE)
    )

  df_stacked <- df %>%
    filter(control_type %in% c("treated", "not-yet treated", "never treated"))

  list(
    df_full    = df,
    df_stacked = df_stacked
  )
}

# ------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------

my_style <- style.df(
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1, " " = 1)
)

make_pretty_dict <- function(min_k = -7, max_k = 3) {
  dict <- c(
    "treated_ig:post_gt"       = "Treated x Post",
    "I(treated_ig * post_gt)"  = "Treated x Post",
    "ATT"                      = "Treated x Post",
    "climate_risk_count"        = "Climate Risk Count",
    "has_climate_risk"           = "Has Climate Risk",
    "climate_risk_pct_unique"    = "% Unique Climate Risks",
    "climate_risk_pct_mentions"  = "% Climate Mentions",
    "mentions_financial_sum"     = "Financial Mentions",
    "mentions_operational_sum"   = "Operational Mentions",
    "mentions_mitigation_sum"    = "Mitigation Mentions",
    "financial_share"            = "Financial Share",
    "operational_share"          = "Operational Share",
    "mitigation_share"           = "Mitigation Share",
    "peer_similarity"            = "Peer Similarity"
  )

  for (k in seq(min_k, max_k)) {
    lab <- paste0("Treated x t", ifelse(k >= 0, paste0("+", k), k))

    dict[paste0("event_time::", k, ":treated_ig")]    <- lab
    dict[paste0("event_time = ", k, " x treated_ig")] <- lab
    dict[paste0("year = ", k)]                         <- lab
  }

  dict
}

# ------------------------------------------------------------------------------
# Sample filtering
# ------------------------------------------------------------------------------

#filter_pre_post <- function(df,
#                            min_pre_years  = 0,
#                            min_post_years = 0) {
#  if (min_pre_years > 0 || min_post_years > 0) {
#    df <- df %>%
#      filter(
#        treatment_type == "never treated" |
#        (
#          treatment_type != "never treated" &
#          (min_pre_years  <= 0 | cnt_pre_years  >= min_pre_years) &
#          (min_post_years <= 0 | cnt_post_years >= min_post_years)
#        )
#      )
#  }
#  df
#}

# ------------------------------------------------------------------------------
# Sun & Abraham (2021) estimators
# ------------------------------------------------------------------------------

make_feols_sunab <- function(df,
                             covariates  = NULL,
                             event_study = TRUE,
                             cluster     = "cik",
                             outcome_var = "climate_risk_count") {

  covars_expr <- substitute(covariates)
  covars_txt  <- if (!is.null(covars_expr) &&
                     !identical(covars_expr, quote(NULL))) {
    deparse1(covars_expr)
  } else { NULL }

  att_value <- if (isTRUE(event_study)) "FALSE" else "TRUE"

  sunab_txt <- paste0(
    "sunab(cohort = year_treated, period = year, no_agg = FALSE, ",
    "ref.c = c(-1000), ref.p = -1, att = ", att_value, ")"
  )

  rhs <- paste(c(na.omit(c(covars_txt, sunab_txt))), collapse = " + ")
  fml <- as.formula(paste0(outcome_var, " ~ ", rhs, " | cik + year"))

  fixest::feols(fml = fml, data = df, cluster = cluster)
}

make_fepois_sunab <- function(df,
                              covariates  = NULL,
                              event_study = TRUE,
                              cluster     = "cik",
                              outcome_var = "climate_risk_count") {

  covars_expr <- substitute(covariates)
  covars_txt  <- if (!is.null(covars_expr) &&
                     !identical(covars_expr, quote(NULL))) {
    deparse1(covars_expr)
  } else { NULL }

  att_value <- if (isTRUE(event_study)) "FALSE" else "TRUE"

  sunab_txt <- paste0(
    "sunab(cohort = year_treated, period = year, no_agg = FALSE, ",
    "ref.c = c(-1000), ref.p = -1, att = ", att_value, ")"
  )

  rhs <- paste(c(na.omit(c(covars_txt, sunab_txt))), collapse = " + ")
  fml <- as.formula(paste0(outcome_var, " ~ ", rhs, " | cik + year"))

  fixest::fepois(fml = fml, data = df, cluster = cluster)
}

make_felogit_sunab <- function(df,
                               covariates  = NULL,
                               event_study = TRUE,
                               cluster     = "cik",
                               outcome_var = "climate_risk_binary") {

  covars_expr <- substitute(covariates)
  covars_txt  <- if (!is.null(covars_expr) &&
                     !identical(covars_expr, quote(NULL))) {
    deparse1(covars_expr)
  } else { NULL }

  att_value <- if (isTRUE(event_study)) "FALSE" else "TRUE"

  sunab_txt <- paste0(
    "sunab(cohort = year_treated, period = year, no_agg = FALSE, ",
    "ref.c = c(-1000), ref.p = -1, att = ", att_value, ")"
  )

  rhs <- paste(c(na.omit(c(covars_txt, sunab_txt))), collapse = " + ")
  fml <- as.formula(paste0(outcome_var, " ~ ", rhs, " | cik + year"))

  fixest::feglm(fml = fml, data = df,
                family = binomial("logit"), cluster = cluster)
}

# ------------------------------------------------------------------------------
# Roth et al. (2023) stacked DiD estimators
# ------------------------------------------------------------------------------

make_fepois_roth <- function(df,
                             covariates  = NULL,
                             event_study = FALSE,
                             cluster     = "cik",
                             outcome_var = "climate_risk_count") {

  covars_expr <- substitute(covariates)
  covars_txt  <- if (!is.null(covars_expr) &&
                     !identical(covars_expr, quote(NULL))) {
    deparse1(covars_expr)
  } else { NULL }

  did_term <- if (isTRUE(event_study)) {
    "i(event_time, treated_ig, ref = -1)"
  } else {
    "treated_ig:post_gt"
  }

  rhs <- paste(c(na.omit(c(covars_txt, did_term))), collapse = " + ")
  fml <- as.formula(paste0(outcome_var, " ~ ", rhs,
                           " | id_group + group_year"))

  fixest::fepois(fml = fml, data = df, cluster = cluster)
}

make_felogit_roth <- function(df,
                              covariates  = NULL,
                              event_study = FALSE,
                              cluster     = "cik",
                              outcome_var = "climate_risk_binary") {

  covars_expr <- substitute(covariates)
  covars_txt  <- if (!is.null(covars_expr) &&
                     !identical(covars_expr, quote(NULL))) {
    deparse1(covars_expr)
  } else { NULL }

  did_term <- if (isTRUE(event_study)) {
    "i(event_time, treated, ref = -1)"
  } else {
    "treated:post"
  }

  rhs <- paste(c(na.omit(c(covars_txt, did_term))), collapse = " + ")
  fml <- as.formula(paste0(outcome_var, " ~ ", rhs,
                           " | id_group + group_year"))

  fixest::feglm(fml = fml, data = df,
                family = binomial("logit"), cluster = cluster)
}

make_feols_roth <- function(df,
                            covariates  = NULL,
                            event_study = FALSE,
                            cluster     = "cik",
                            outcome_var = "climate_risk_count") {

  covars_expr <- substitute(covariates)
  covars_txt  <- if (!is.null(covars_expr) &&
                     !identical(covars_expr, quote(NULL))) {
    deparse1(covars_expr)
  } else { NULL }

  did_term <- if (isTRUE(event_study)) {
    "i(event_time, treated, ref = -1)"
  } else {
    "treated:post"
  }

  rhs <- paste(c(na.omit(c(covars_txt, did_term))), collapse = " + ")
  fml <- as.formula(paste0(outcome_var, " ~ ", rhs,
                           " | id_group + group_year"))

  fixest::feols(fml = fml, data = df, cluster = cluster)
}

# ------------------------------------------------------------------------------
# Lee (2009) trimming bounds — robustness for intensive margin
# ------------------------------------------------------------------------------

compute_trimming_q <- function(df,
                               treated_var    = "treated",
                               event_time_var = "event_time",
                               selection_var  = "has_climate_risk") {

  p1 <- df %>%
    filter(.data[[treated_var]] == 1,
           !is.na(.data[[event_time_var]]),
           .data[[event_time_var]] >= 0) %>%
    summarise(rate = mean(.data[[selection_var]], na.rm = TRUE)) %>%
    pull(rate)

  p0 <- df %>%
    filter(.data[[treated_var]] == 0) %>%
    summarise(rate = mean(.data[[selection_var]], na.rm = TRUE)) %>%
    pull(rate)

  list(q = 1 - p0 / p1, p_treated_post = p1, p_control = p0)
}

lee_trim <- function(df, outcome_var, q,
                     treated_var    = "treated",
                     event_time_var = "event_time",
                     selection_var  = "has_climate_risk") {

  df_disc <- df %>% filter(.data[[selection_var]] == 1)

  is_tp <- df_disc[[treated_var]] == 1 &
    !is.na(df_disc[[event_time_var]]) &
    df_disc[[event_time_var]] >= 0

  y       <- df_disc[[outcome_var]]
  y_valid <- y[is_tp & !is.na(y)]

  cut_lo  <- quantile(y_valid, probs = q,     na.rm = TRUE)
  cut_hi  <- quantile(y_valid, probs = 1 - q, na.rm = TRUE)

  keep_ub <- !is_tp | is.na(y) | y >= cut_lo
  keep_lb <- !is_tp | is.na(y) | y <= cut_hi

  list(
    naive          = df_disc,
    upper          = df_disc[keep_ub, ],
    lower          = df_disc[keep_lb, ],
    n_disc         = nrow(df_disc),
    n_treated_post = sum(is_tp),
    n_trimmed      = sum(is_tp & !is.na(y)) - sum(keep_ub & is_tp & !is.na(y)),
    cut_lo         = cut_lo,
    cut_hi         = cut_hi
  )
}

extract_sunab_coefs <- function(model, label) {
  cf  <- coeftable(model)
  idx <- grep("^year", rownames(cf))
  if (length(idx) == 0) return(data.frame())

  cf_sub <- cf[idx, , drop = FALSE]
  et     <- as.numeric(gsub(".*= ?([-0-9]+).*", "\\1", rownames(cf_sub)))

  data.frame(
    event_time = et,
    estimate   = cf_sub[, 1],
    se         = cf_sub[, 2],
    bound      = label,
    row.names  = NULL
  ) %>%
    mutate(ci_lo = estimate - 1.96 * se,
           ci_hi = estimate + 1.96 * se)
}

# ------------------------------------------------------------------------------
# de Chaisemartin & D'Haultfoeuille (2024) estimator
# ------------------------------------------------------------------------------

#' Prepare multi-event panel for dCDH estimation.
#' Constructs binary absorbing treatment D and deduplicates by (cik, year).
#prepare_dcdh <- function(parquet_path, exclude_ciks = NULL) {
#  df <- read_parquet(parquet_path)

#  if (!is.null(exclude_ciks)) {
#    df <- df %>% filter(!cik %in% exclude_ciks)
#  }

#  df <- df %>%
#    distinct(cik, year, .keep_all = TRUE) %>%
#    mutate(
#      cik_num = as.numeric(cik),
#      D       = as.numeric(!is.na(first_event_year) & year >= first_event_year),
#      log_climate_risk_count = log(1 + climate_risk_count)
#    )

#  cat(sprintf("Panel: %s obs | %d firms | years %d-%d\n",
#              format(nrow(df), big.mark = ","), n_distinct(df$cik),
#              min(df$year), max(df$year)))
#  cat(sprintf("D=0: %s (%.1f%%) | D=1: %s (%.1f%%) | Switchers: %d\n",
#              format(sum(df$D == 0), big.mark = ","), 100 * mean(df$D == 0),
#              format(sum(df$D == 1), big.mark = ","), 100 * mean(df$D == 1),
#              n_distinct(df$cik[df$D == 1])))

#  df
#}

#' Run dCDH estimation for all 5 outcome specifications.
run_dcdh_specs <- function(df, effects = 3, placebo = 3, specs = NULL) {

  results <- list()
  for (s in specs) {
    cat(sprintf("  %s ... ", s$label))
    results[[s$label]] <- did_multiplegt_dyn(
      df        = as.data.frame(df),
      outcome   = s$outcome,
      group     = "cik_num",
      time      = "year",
      treatment = "D",
      effects   = effects,
      placebo   = placebo,
      cluster   = "cik_num",
      graph_off = TRUE
    )
    cat(sprintf("ATT = %.4f (SE = %.4f)\n",
                results[[s$label]]$results$ATE[1, 1],
                results[[s$label]]$results$ATE[1, 2]))
  }

  results
}

#' Format dCDH results into a regression-style table.
build_dcdh_table <- function(results, n_obs = NULL) {
  add_stars <- function(est, se) {
    z <- abs(est / se)
    stars <- ifelse(z > 2.576, "***", ifelse(z > 1.96, "**",
                    ifelse(z > 1.645, "*", "")))
    sprintf("%.4f%s", est, stars)
  }
  format_se <- function(se) sprintf("(%.4f)", se)

  row_labels <- c("Placebo 3", "Placebo 2", "Placebo 1",
                   "Effect 1", "Effect 2", "Effect 3",
                   "ATT", "", "N", "Switchers",
                   "Joint placebo p", "Joint effects p")

  out <- tibble(` ` = row_labels)

  for (nm in names(results)) {
    res <- results[[nm]]
    eff <- as.data.frame(res$results$Effects)
    pl  <- as.data.frame(res$results$Placebos)
    ate <- as.data.frame(res$results$ATE)

    total_n <- ifelse(!is.null(n_obs), n_obs, sum(eff[, "N"]))

    att_col <- c(
      rep("", 6),
      add_stars(ate[1, 1], ate[1, 2]),
      format_se(ate[1, 2]),
      format(total_n, big.mark = ","),
      format(ate[1, "Switchers"], big.mark = ","),
      "", ""
    )

    es_est <- c(rev(pl[, 1]), eff[, 1])
    es_se  <- c(rev(pl[, 2]), eff[, 2])

    smry  <- capture.output(summary(res))
    p_pl  <- as.numeric(sub(".*p-value = ", "",
                            grep("placebos.*p-value", smry, value = TRUE)))
    p_eff <- as.numeric(sub(".*p-value = ", "",
                            grep("effects.*p-value", smry, value = TRUE)))

    es_col <- c(
      mapply(add_stars, es_est, es_se),
      add_stars(ate[1, 1], ate[1, 2]),
      format_se(ate[1, 2]),
      format(total_n, big.mark = ","),
      format(ate[1, "Switchers"], big.mark = ","),
      sprintf("%.4f", p_pl),
      sprintf("%.4f", p_eff)
    )

    out[[paste0(nm, " ATT")]] <- att_col
    out[[paste0(nm, " ES")]]  <- es_col
  }

  out
}

#' Build Restriction A sample: keep only switchers with >= min_pre
#' pre-treatment periods, plus all never-treated firms.
build_dcdh_restriction_a <- function(df, min_pre = 3) {
  dt <- data.table::as.data.table(df)

  pre_periods <- dt[!is.na(first_event_year) & year < first_event_year,
                    .N, by = cik_num]
  data.table::setnames(pre_periods, c("cik_num", "n_pre"))
  cik_keep <- pre_periods[n_pre >= min_pre, cik_num]

  dt_a <- dt[cik_num %in% cik_keep | is.na(first_event_year)]
  dt_a$log_climate_risk_count <- log(1 + dt_a$climate_risk_count)

  n_sw <- data.table::uniqueN(dt_a[!is.na(first_event_year), cik_num])
  n_nt <- data.table::uniqueN(dt_a[is.na(first_event_year), cik_num])
  cat(sprintf("Restriction A (>= %d pre-periods): %d firms (%d switchers, %d never-treated) | %s obs\n",
              min_pre, data.table::uniqueN(dt_a$cik_num), n_sw, n_nt,
              format(nrow(dt_a), big.mark = ",")))

  as.data.frame(dt_a)
}
