#!/usr/bin/env python3
"""
Regenerate the key numeric tables for the HW1 momentum report.

This script mirrors the notebook's exact portfolio construction and regression
definitions (EW / VW / RW; CAPM regression; FF3 regression; Sharpe).

Outputs are written as LaTeX tabular fragments into:
  hw1/docs/hw1_report_tables/
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm


def load_data(repo_root: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    data_dir = repo_root / "hw1" / "data"

    ret_df = pl.scan_csv(data_dir / "returns_indiv.csv").collect()
    ff_df = pl.scan_csv(data_dir / "monthly_ff3.csv").collect()

    # Match the notebook: convert FF first column (unnamed) from YYYYMM to month-end date.
    ff_date_col = ff_df.columns[0]
    ff_df = (
        ff_df.with_columns(
            pl.col(ff_date_col)
            .cast(str)
            .str.to_date("%Y%m")
            .dt.month_end()
            .alias("date"),
            pl.col("Mkt-RF").str.strip_chars().cast(pl.Float64),
            pl.col("SMB").str.strip_chars().cast(pl.Float64),
            pl.col("HML").str.strip_chars().cast(pl.Float64),
            pl.col("RF").str.strip_chars().cast(pl.Float64),
        )
        .select("date", "Mkt-RF", "SMB", "HML", "RF")
        .sort("date")
    )

    # Match the notebook: convert returns date from YYYYMMDD to month-end date.
    ret_df = ret_df.with_columns(
        pl.col("date").cast(str).str.to_date("%Y%m%d").dt.month_end().alias("date")
    ).sort(["date", "permno"])

    return ret_df, ff_df


def construct_momentum_12_1(df: pl.DataFrame) -> pl.DataFrame:
    # Notebook definition: exp(sum(log(1+ret_k))) - 1 for k=1..12
    df = df.with_columns(
        (
            pl.sum_horizontal((pl.col([f"ret_{i}" for i in range(1, 13)]) + 1).log()).exp()
            - 1
        ).alias("mom_12_1")
    )
    return df.select("permno", "date", "mom_12_1")


def _assign_deciles(df: pl.DataFrame, signal_name: str) -> pl.DataFrame:
    df = df.with_columns(pl.col(signal_name).rank(method="average").over("date").alias("rank"))
    df = df.with_columns(
        ((pl.col("rank") - 1) / pl.col("rank").count().over("date") * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .add(1)
        .alias("decile")
    )
    return df


def construct_decile_ew_portfolios(
    signal_df: pl.DataFrame, signal_name: str, return_df: pl.DataFrame
) -> pl.DataFrame:
    df = signal_df.join(
        return_df.select("permno", "date", "ret_p1"),
        on=["permno", "date"],
        how="left",
        validate="1:1",
    )
    df = _assign_deciles(df, signal_name)
    return (
        df.group_by(["date", "decile"])
        .agg(
            pl.col("ret_p1").mean().alias("portfolio_return"),
            pl.len().alias("n_stocks"),
        )
        .sort(["date", "decile"])
    )


def construct_decile_vw_portfolios(
    signal_df: pl.DataFrame, signal_name: str, return_df: pl.DataFrame
) -> pl.DataFrame:
    df = signal_df.join(
        return_df.select("permno", "date", "ret_p1", "market_cap"),
        on=["permno", "date"],
        how="left",
        validate="1:1",
    )
    df = _assign_deciles(df, signal_name)
    return (
        df.group_by(["date", "decile"])
        .agg(
            (
                pl.col("ret_p1") * (pl.col("market_cap") / pl.col("market_cap").sum())
            )
            .sum()
            .alias("portfolio_return"),
            pl.len().alias("n_stocks"),
        )
        .sort(["date", "decile"])
    )


def construct_decile_rw_portfolios(
    signal_df: pl.DataFrame, signal_name: str, return_df: pl.DataFrame
) -> pl.DataFrame:
    df = signal_df.join(
        return_df.select("permno", "date", "ret_p1"),
        on=["permno", "date"],
        how="left",
        validate="1:1",
    )
    df = _assign_deciles(df, signal_name)
    df = df.with_columns(
        (pl.col("rank") / pl.col("rank").sum().over(["date", "decile"])).alias("weight")
    )
    return (
        df.group_by(["date", "decile"])
        .agg(
            (pl.col("weight") * pl.col("ret_p1")).sum().alias("portfolio_return"),
            pl.len().alias("n_stocks"),
        )
        .sort(["date", "decile"])
    )


def run_capm_regression(portfolio_returns_df: pl.DataFrame, ff_df: pl.DataFrame) -> pd.DataFrame:
    df = portfolio_returns_df.join(
        ff_df.select("date", "Mkt-RF", "RF"),
        on="date",
        how="left",
        validate="m:1",
    ).with_columns((pl.col("portfolio_return") * 100 - pl.col("RF")).alias("excess_return"))

    df_pd = df.select("decile", "excess_return", "Mkt-RF").to_pandas()

    def capm_regression(group: pd.DataFrame) -> pd.Series:
        y = group["excess_return"].values
        X = sm.add_constant(group["Mkt-RF"].values)
        model = sm.OLS(y, X, missing="drop").fit()
        return pd.Series(
            {
                "alpha": model.params[0],
                "beta": model.params[1],
                "se_alpha": model.bse[0],
                "se_beta": model.bse[1],
                "t_alpha": model.tvalues[0],
                "t_beta": model.tvalues[1],
            }
        )

    return df_pd.groupby("decile").apply(capm_regression, include_groups=False)


def construct_long_short_portfolio(portfolio_returns_df: pl.DataFrame) -> pl.DataFrame:
    pivoted = portfolio_returns_df.pivot(
        index="date", on="decile", values="portfolio_return", aggregate_function=None
    )
    return pivoted.select(pl.col("date"), (pl.col("10") - pl.col("1")).alias("ls_return"))


def calculate_ls_statistics(ls_df: pl.DataFrame, ff_df: pl.DataFrame) -> Dict[str, float]:
    df = ls_df.join(
        ff_df.select("date", "Mkt-RF", "SMB", "HML", "RF"), on="date", how="inner"
    ).with_columns(
        (pl.col("ls_return") * 100 - pl.col("RF")).alias("excess_return"),
        (pl.col("ls_return") * 100).alias("raw_return"),
    )

    df_pd = df.to_pandas()

    avg_raw_return_monthly = df_pd["raw_return"].mean()
    avg_raw_return_annualized = avg_raw_return_monthly * 12

    y_capm = df_pd["excess_return"].values
    X_capm = sm.add_constant(df_pd["Mkt-RF"].values)
    capm = sm.OLS(y_capm, X_capm, missing="drop").fit()

    y_ff = df_pd["excess_return"].values
    X_ff = sm.add_constant(df_pd[["Mkt-RF", "SMB", "HML"]].values)
    ff = sm.OLS(y_ff, X_ff, missing="drop").fit()

    mean_excess = df_pd["excess_return"].mean()
    std_excess = df_pd["excess_return"].std()
    annual_sharpe = (mean_excess / std_excess) * math.sqrt(12) if std_excess != 0 else np.nan

    return {
        "avg_raw_return_annualized_pct": float(avg_raw_return_annualized),
        "capm_alpha": float(capm.params[0]),
        "capm_alpha_tstat": float(capm.tvalues[0]),
        "ff_alpha": float(ff.params[0]),
        "ff_alpha_tstat": float(ff.tvalues[0]),
        "annual_sharpe": float(annual_sharpe),
    }


def construct_tau_horizon_ls_portfolio(
    signal_df: pl.DataFrame,
    signal_name: str,
    return_df: pl.DataFrame,
    tau: int,
    weighting: str = "ew",
) -> pl.DataFrame:
    ret_col = f"ret_p{tau}"

    if weighting == "ew":
        df = signal_df.join(
            return_df.select("permno", "date", ret_col),
            on=["permno", "date"],
            how="left",
            validate="1:1",
        )
    elif weighting == "vw":
        df = signal_df.join(
            return_df.select("permno", "date", ret_col, "market_cap"),
            on=["permno", "date"],
            how="left",
            validate="1:1",
        )
    elif weighting == "rw":
        df = signal_df.join(
            return_df.select("permno", "date", ret_col),
            on=["permno", "date"],
            how="left",
            validate="1:1",
        )
    else:
        raise ValueError(f"Unknown weighting={weighting!r}; expected one of: ew, vw, rw")

    df = _assign_deciles(df, signal_name)

    if weighting == "ew":
        port = (
            df.group_by(["date", "decile"])
            .agg(pl.col(ret_col).mean().alias("portfolio_return"))
            .sort(["date", "decile"])
        )
    elif weighting == "vw":
        port = (
            df.group_by(["date", "decile"])
            .agg(
                (pl.col(ret_col) * (pl.col("market_cap") / pl.col("market_cap").sum()))
                .sum()
                .alias("portfolio_return")
            )
            .sort(["date", "decile"])
        )
    else:  # rw
        df = df.with_columns(
            (pl.col("rank") / pl.col("rank").sum().over(["date", "decile"])).alias("weight")
        )
        port = (
            df.group_by(["date", "decile"])
            .agg((pl.col("weight") * pl.col(ret_col)).sum().alias("portfolio_return"))
            .sort(["date", "decile"])
        )

    pivoted = port.pivot(index="date", on="decile", values="portfolio_return", aggregate_function=None)
    return pivoted.select(pl.col("date"), (pl.col("10") - pl.col("1")).alias("ls_return"))


def calculate_ff_alpha_for_horizon(ls_df: pl.DataFrame, ff_df: pl.DataFrame) -> Tuple[float, float]:
    df = ls_df.join(ff_df.select("date", "Mkt-RF", "SMB", "HML", "RF"), on="date", how="inner")
    df = df.with_columns((pl.col("ls_return") * 100 - pl.col("RF")).alias("excess_return"))
    df_pd = df.to_pandas().dropna()
    if len(df_pd) < 10:
        return float("nan"), float("nan")
    y = df_pd["excess_return"].values
    X = sm.add_constant(df_pd[["Mkt-RF", "SMB", "HML"]].values)
    model = sm.OLS(y, X, missing="drop").fit()
    return float(model.params[0]), float(model.tvalues[0])


def _write_latex_table(df: pd.DataFrame, out_path: Path, float_fmt: str = "%.4f") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def esc(s: str) -> str:
        # IMPORTANT: we intentionally do NOT escape backslashes here so that
        # LaTeX in labels (e.g., \%, \alpha, $t$) passes through correctly.
        return (
            s.replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
            .replace("#", "\\#")
        )

    def fmt(x) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        if isinstance(x, (float, np.floating, int, np.integer)):
            return float_fmt % float(x)
        return esc(str(x))

    index_name = df.index.name or ""
    col_names = [str(c) for c in df.columns]

    # Column alignment: index left, data right
    col_spec = "l" + ("r" * len(col_names))

    lines: List[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header_cells = [esc(index_name)] + [esc(c) for c in col_names]
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    for idx, row in df.iterrows():
        row_cells = [esc(str(idx))] + [fmt(row[c]) for c in df.columns]
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "hw1" / "docs" / "hw1_report_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ret_df, ff_df = load_data(repo_root)
    mom_df = construct_momentum_12_1(ret_df)

    ew = construct_decile_ew_portfolios(mom_df, "mom_12_1", ret_df)
    vw = construct_decile_vw_portfolios(mom_df, "mom_12_1", ret_df)
    rw = construct_decile_rw_portfolios(mom_df, "mom_12_1", ret_df)

    # Q1: average monthly raw return by decile (in %)
    def avg_by_decile(port: pl.DataFrame) -> pd.Series:
        s = (
            port.group_by("decile")
            .agg(pl.col("portfolio_return").mean().alias("mean_ret"))
            .sort("decile")
            .to_pandas()
            .set_index("decile")["mean_ret"]
        )
        return s * 100.0

    q1_df = pd.DataFrame(
        {
            "EW (\\%)": avg_by_decile(ew),
            "VW (\\%)": avg_by_decile(vw),
            "RW (\\%)": avg_by_decile(rw),
        }
    )
    q1_df.index.name = "Decile"
    _write_latex_table(q1_df, out_dir / "q1_decile_raw_returns.tex", float_fmt="%.3f")

    # Q2: CAPM per decile
    capm_ew = run_capm_regression(ew, ff_df)
    capm_vw = run_capm_regression(vw, ff_df)
    capm_rw = run_capm_regression(rw, ff_df)
    _write_latex_table(capm_ew, out_dir / "q2_capm_ew.tex", float_fmt="%.4f")
    _write_latex_table(capm_vw, out_dir / "q2_capm_vw.tex", float_fmt="%.4f")
    _write_latex_table(capm_rw, out_dir / "q2_capm_rw.tex", float_fmt="%.4f")

    # Q3: long-short stats
    ew_ls = construct_long_short_portfolio(ew)
    vw_ls = construct_long_short_portfolio(vw)
    rw_ls = construct_long_short_portfolio(rw)
    ls_stats = pd.DataFrame(
        {
            "Equal-Weighted": calculate_ls_statistics(ew_ls, ff_df),
            "Value-Weighted": calculate_ls_statistics(vw_ls, ff_df),
            "Rank-Weighted": calculate_ls_statistics(rw_ls, ff_df),
        }
    ).T
    ls_stats = ls_stats.rename(
        columns={
            "avg_raw_return_annualized_pct": "Avg raw (ann., \\%)",
            "capm_alpha": "CAPM $\\alpha$",
            "capm_alpha_tstat": "$t$ (CAPM $\\alpha$)",
            "ff_alpha": "FF3 $\\alpha$",
            "ff_alpha_tstat": "$t$ (FF3 $\\alpha$)",
            "annual_sharpe": "Sharpe (ann.)",
        }
    )
    _write_latex_table(ls_stats, out_dir / "q3_longshort_summary.tex", float_fmt="%.4f")

    # Q4: horizons τ=1..36 (FF alpha and t-stat) for EW / VW / RW, plus a small summary table.
    horizons = list(range(1, 37))
    ew_rows: List[Dict[str, float]] = []
    vw_rows: List[Dict[str, float]] = []
    rw_rows: List[Dict[str, float]] = []
    for tau in horizons:
        ew_tau_ls = construct_tau_horizon_ls_portfolio(mom_df, "mom_12_1", ret_df, tau, "ew")
        vw_tau_ls = construct_tau_horizon_ls_portfolio(mom_df, "mom_12_1", ret_df, tau, "vw")
        rw_tau_ls = construct_tau_horizon_ls_portfolio(mom_df, "mom_12_1", ret_df, tau, "rw")
        ew_a, ew_t = calculate_ff_alpha_for_horizon(ew_tau_ls, ff_df)
        vw_a, vw_t = calculate_ff_alpha_for_horizon(vw_tau_ls, ff_df)
        rw_a, rw_t = calculate_ff_alpha_for_horizon(rw_tau_ls, ff_df)
        ew_rows.append({"tau": tau, "ff_alpha": ew_a, "t_stat": ew_t})
        vw_rows.append({"tau": tau, "ff_alpha": vw_a, "t_stat": vw_t})
        rw_rows.append({"tau": tau, "ff_alpha": rw_a, "t_stat": rw_t})

    ew_h = pd.DataFrame(ew_rows).set_index("tau")
    vw_h = pd.DataFrame(vw_rows).set_index("tau")
    rw_h = pd.DataFrame(rw_rows).set_index("tau")
    _write_latex_table(ew_h, out_dir / "q4_horizon_ew_full.tex", float_fmt="%.4f")
    _write_latex_table(vw_h, out_dir / "q4_horizon_vw_full.tex", float_fmt="%.4f")
    _write_latex_table(rw_h, out_dir / "q4_horizon_rw_full.tex", float_fmt="%.4f")

    # Small “representative horizons” table for the report body
    taus_small = [1, 3, 6, 12, 24, 36]
    q4_small = pd.DataFrame(
        {
            "EW $\\alpha$": ew_h.loc[taus_small, "ff_alpha"],
            "EW $t$": ew_h.loc[taus_small, "t_stat"],
            "VW $\\alpha$": vw_h.loc[taus_small, "ff_alpha"],
            "VW $t$": vw_h.loc[taus_small, "t_stat"],
            "RW $\\alpha$": rw_h.loc[taus_small, "ff_alpha"],
            "RW $t$": rw_h.loc[taus_small, "t_stat"],
        },
        index=taus_small,
    )
    q4_small.index.name = "tau"
    _write_latex_table(q4_small, out_dir / "q4_horizon_small.tex", float_fmt="%.4f")

    print(f"Wrote LaTeX tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

