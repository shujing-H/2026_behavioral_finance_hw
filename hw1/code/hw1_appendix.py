"""
Auto-generated from hw1/code/hw1.ipynb.

This file is intended for the homework report appendix.
"""

# flake8: noqa

# %% [cell 1]
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# %% [cell 2]
# read data
ret_df = pl.scan_csv("../data/returns_indiv.csv").collect()
ff_factors_df = pl.scan_csv("../data/monthly_ff3.csv").collect()

print(ret_df.head(2))
print(ff_factors_df.head(2))

# %% [cell 3]
ret_df["market_cap"].describe()

# %% [cell 4]
# change the dtypes for ff_factors_df
ff_factors_df = ff_factors_df.with_columns(
    pl.col("").cast(str).str.to_date("%Y%m").dt.month_end().alias("date"),
    pl.col("Mkt-RF").str.strip_chars().cast(pl.Float64),
    pl.col("SMB").str.strip_chars().cast(pl.Float64),
    pl.col("HML").str.strip_chars().cast(pl.Float64),
    pl.col("RF").str.strip_chars().cast(pl.Float64),
).select("date", "Mkt-RF", "SMB", "HML", "RF")

# %% [cell 5]
# change the dates in ret_df into date
ret_df = ret_df.with_columns(
    pl.col("date").cast(str).str.to_date("%Y%m%d").dt.month_end().alias("date"),
)

# %% [cell 6]
ret_df.columns

# %% [cell 7]
ret_df.select(
    (
        pl.sum_horizontal(
            (pl.col(["ret_" + str(i) for i in range(1, 13)]) + 1).log()
        ).exp()
        - 1
    ).alias("mom_12_1")
).sort("mom_12_1", descending=True)

# %% [cell 8]
def construct_momentum_12_1(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construct 12-1 momentum signal:
        calculate the cumulative return of from month t-12 to t-1 (no prod_horizontal expr in polars, we first transform the returns into log returns, then sum them up, and then take the exponential of the sum)
    """
    df = df.with_columns(
        (
            pl.sum_horizontal(
                (pl.col(["ret_" + str(i) for i in range(1, 13)]) + 1).log()
            ).exp()
            - 1
        ).alias("mom_12_1")
    )
    return df.select("permno", "date", "mom_12_1")


def construct_decile_ew_portfolios(
    signal_df: pl.DataFrame, signal_name: str, return_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Construct 10 decile equal-weighted portfolios based on month-t momentum, hold these portfolios for 1 month, and then rebalance.
    Return the portfolio returns.
    """
    # Join signal with forward returns
    df = signal_df.join(
        return_df.select("permno", "date", "ret_p1"),
        on=["permno", "date"],
        how="left",
        validate="1:1",
    )

    # Rank signals within each date and assign to deciles
    df = df.with_columns(
        pl.col(signal_name).rank(method="average").over("date").alias("rank")
    )

    # Calculate decile assignments (1-10)
    df = df.with_columns(
        ((pl.col("rank") - 1) / pl.col("rank").count().over("date") * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .add(1)
        .alias("decile")
    )

    # Calculate equal-weighted portfolio returns for each decile
    portfolio_returns = (
        df.group_by(["date", "decile"])
        .agg(
            pl.col("ret_p1").mean().alias("portfolio_return"),
            pl.len().alias("n_stocks"),
        )
        .sort(["date", "decile"])
    )

    return portfolio_returns


def construct_decile_vw_portfolios(
    signal_df: pl.DataFrame, signal_name: str, return_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Construct 10 decile value-weighted portfolios based on month-t momentum, hold these portfolios for 1 month, and then rebalance.
    Return the portfolio returns.
    """
    # Join signal with forward returns
    df = signal_df.join(
        return_df.select("permno", "date", "ret_p1", "market_cap"),
        on=["permno", "date"],
        how="left",
        validate="1:1",
    )

    # Rank signals within each date and assign to deciles
    df = df.with_columns(
        pl.col(signal_name).rank(method="average").over("date").alias("rank")
    )

    # Calculate decile assignments (1-10)
    df = df.with_columns(
        ((pl.col("rank") - 1) / pl.col("rank").count().over("date") * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .add(1)
        .alias("decile")
    )

    # Calculate value-weighted portfolio returns for each decile
    portfolio_returns = (
        df.group_by(["date", "decile"])
        .agg(
            (pl.col("ret_p1") * (pl.col("market_cap") / pl.col("market_cap").sum()))
            .sum()
            .alias("portfolio_return"),
            pl.len().alias("n_stocks"),
        )
        .sort(["date", "decile"])
    )

    return portfolio_returns


def construct_decile_rw_portfolios(
    signal_df: pl.DataFrame, signal_name: str, return_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Construct 10 decile rank-weighted portfolios based on month-t momentum.
    Weights are proportional to the rank of the momentum signal (descending:
    larger mom_12_1 gets larger weight). Hold for 1 month, then rebalance.
    Return the portfolio returns.
    """
    # Join signal with forward returns
    df = signal_df.join(
        return_df.select("permno", "date", "ret_p1"),
        on=["permno", "date"],
        how="left",
        validate="1:1",
    )

    # Rank signals within each date (ascending: low mom=1, high mom=N, so larger mom=larger weight)
    df = df.with_columns(
        pl.col(signal_name).rank(method="average").over("date").alias("rank")
    )

    # Calculate decile assignments (1-10)
    df = df.with_columns(
        ((pl.col("rank") - 1) / pl.col("rank").count().over("date") * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .add(1)
        .alias("decile")
    )

    # Rank-weighted: weight = rank / sum(rank) within each decile
    df = df.with_columns(
        (pl.col("rank") / pl.col("rank").sum().over(["date", "decile"])).alias("weight")
    )

    # Calculate rank-weighted portfolio returns for each decile
    portfolio_returns = (
        df.group_by(["date", "decile"])
        .agg(
            (pl.col("weight") * pl.col("ret_p1")).sum().alias("portfolio_return"),
            pl.len().alias("n_stocks"),
        )
        .sort(["date", "decile"])
    )

    return portfolio_returns


def plot_decile_portfolios(
    portfolio_returns_df: pl.DataFrame, signal_name: str
) -> pl.DataFrame:
    """
    Plot the decile portfolios cumulative returns.
    Args:
        portfolio_returns_df: pl.DataFrame, the portfolio returns.
        signal_name: str, the name of the signal.
    """
    # Calculate cumulative returns
    portfolio_returns_df = portfolio_returns_df.sort("date").with_columns(
        (1 + pl.col("portfolio_return"))
        .cum_prod()
        .over("decile")
        .alias("cumulative_return")
    )

    # Plot cumulative returns
    portfolio_returns_df_pd = (
        portfolio_returns_df.pivot(
            index="date",
            on="decile",
            values="cumulative_return",
            aggregate_function=None,
        )
        .with_columns(
            (pl.col("10") - pl.col("1")).alias("long_short"),
        )
        .to_pandas()
        .set_index("date")
    )

    portfolio_returns_df_pd.iloc[:, :-1].plot(figsize=(10, 6), linestyle="--")
    portfolio_returns_df_pd.iloc[:, -1].plot(figsize=(10, 6), color="black")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.title(f"Cumulative Returns of Decile Portfolios for {signal_name}")
    plt.show()

    return pl.DataFrame(portfolio_returns_df_pd.reset_index())

# %% [cell 9]
mom_12_1_df = construct_momentum_12_1(ret_df)
ew_portfolio_returns_df = construct_decile_ew_portfolios(
    mom_12_1_df, "mom_12_1", ret_df
)
vw_portfolio_returns_df = construct_decile_vw_portfolios(
    mom_12_1_df, "mom_12_1", ret_df
)
rw_portfolio_returns_df = construct_decile_rw_portfolios(
    mom_12_1_df, "mom_12_1", ret_df
)
ew_portfolio_returns_df_pivoted = plot_decile_portfolios(
    ew_portfolio_returns_df, "mom_12_1 (ew)"
)
vw_portfolio_returns_df_pivoted = plot_decile_portfolios(
    vw_portfolio_returns_df, "mom_12_1 (vw)"
)
rw_portfolio_returns_df_pivoted = plot_decile_portfolios(
    rw_portfolio_returns_df, "mom_12_1 (rw)"
)

# %% [cell 10]
# plot three long-short portfolios
long_short_df = (
    ew_portfolio_returns_df_pivoted.select(
        "date", pl.col("long_short").alias("long_short_ew")
    )
    .join(
        vw_portfolio_returns_df_pivoted.select(
            "date", pl.col("long_short").alias("long_short_vw")
        ),
        on="date",
        how="left",
        validate="1:1",
    )
    .join(
        rw_portfolio_returns_df_pivoted.select(
            "date", pl.col("long_short").alias("long_short_rw")
        ),
        on="date",
        how="left",
        validate="1:1",
    )
    .to_pandas()
    .set_index("date")
    .sort_index()
    .plot(figsize=(10, 6))
)

# %% [cell 11]
def run_capm_regression(
    portfolio_returns_df: pl.DataFrame, mkt_factor_df: pl.DataFrame
) -> pd.DataFrame:
    """
    Run the CAPM regression for each portfolio decile.
    Uses groupby.apply to avoid explicit for-loops.
    """
    # Join portfolio returns with market factor (keep long format)
    df = portfolio_returns_df.join(
        mkt_factor_df.select("date", "Mkt-RF", "RF"),
        on="date",
        how="left",
        validate="m:1",
    )

    # Compute excess returns (portfolio return - RF)
    df = df.with_columns(
        (pl.col("portfolio_return") * 100 - pl.col("RF")).alias("excess_return")
    )

    # Convert to pandas for groupby regression
    df_pd = df.select("decile", "excess_return", "Mkt-RF").to_pandas()

    def capm_regression(group: pd.DataFrame) -> pd.Series:
        """Run CAPM regression on a single group."""
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

    # Run regression per decile using groupby.apply (no explicit for-loop)
    results = df_pd.groupby("decile").apply(capm_regression, include_groups=False)

    return results

# %% [cell 12]
# Run CAPM regression for EW and VW portfolios
ew_capm_results = run_capm_regression(ew_portfolio_returns_df, ff_factors_df)
vw_capm_results = run_capm_regression(vw_portfolio_returns_df, ff_factors_df)
rw_capm_results = run_capm_regression(rw_portfolio_returns_df, ff_factors_df)

print("Equal-Weighted Portfolios CAPM Results:")
display(ew_capm_results.round(4))
print("\nValue-Weighted Portfolios CAPM Results:")
display(vw_capm_results.round(4))
print("\nRank-Weighted Portfolios CAPM Results:")
display(rw_capm_results.round(4))

# %% [cell 13]
def construct_long_short_portfolio(portfolio_returns_df: pl.DataFrame) -> pl.DataFrame:
    """
    Construct a long-short portfolio: long top decile (10), short bottom decile (1).
    """
    # Pivot to get decile returns side by side
    pivoted = portfolio_returns_df.pivot(
        index="date",
        on="decile",
        values="portfolio_return",
        aggregate_function=None,
    )

    # Long-short return = decile 10 - decile 1
    ls_df = pivoted.select(
        pl.col("date"),
        (pl.col("10") - pl.col("1")).alias("ls_return"),
    )

    return ls_df


def calculate_ls_statistics(ls_df: pl.DataFrame, ff_factors_df: pl.DataFrame) -> dict:
    """
    Calculate long-short portfolio statistics:
    - Average raw return (annualized)
    - CAPM alpha and t-stat
    - Fama-French 3-factor alpha and t-stat
    - Annual Sharpe ratio
    """
    # Join with FF factors
    df = ls_df.join(
        ff_factors_df.select("date", "Mkt-RF", "SMB", "HML", "RF"),
        on="date",
        how="inner",
    )

    # Compute excess return (convert to percent to match FF factors)
    df = df.with_columns(
        (pl.col("ls_return") * 100 - pl.col("RF")).alias("excess_return"),
        (pl.col("ls_return") * 100).alias("raw_return"),
    )

    df_pd = df.to_pandas()

    # Average raw return (monthly, in percent) - annualized
    avg_raw_return_monthly = df_pd["raw_return"].mean()
    avg_raw_return_annualized = avg_raw_return_monthly * 12

    # CAPM regression: excess_return = alpha + beta * Mkt-RF + epsilon
    y_capm = df_pd["excess_return"].values
    X_capm = sm.add_constant(df_pd["Mkt-RF"].values)
    capm_model = sm.OLS(y_capm, X_capm, missing="drop").fit()
    capm_alpha = capm_model.params[0]
    capm_alpha_tstat = capm_model.tvalues[0]

    # Fama-French 3-factor regression
    y_ff = df_pd["excess_return"].values
    X_ff = sm.add_constant(df_pd[["Mkt-RF", "SMB", "HML"]].values)
    ff_model = sm.OLS(y_ff, X_ff, missing="drop").fit()
    ff_alpha = ff_model.params[0]
    ff_alpha_tstat = ff_model.tvalues[0]

    # Annual Sharpe ratio: (mean excess return / std excess return) * sqrt(12)
    mean_excess = df_pd["excess_return"].mean()
    std_excess = df_pd["excess_return"].std()
    annual_sharpe = (mean_excess / std_excess) * np.sqrt(12)

    return {
        "avg_raw_return_annualized_pct": avg_raw_return_annualized,
        "capm_alpha": capm_alpha,
        "capm_alpha_tstat": capm_alpha_tstat,
        "ff_alpha": ff_alpha,
        "ff_alpha_tstat": ff_alpha_tstat,
        "annual_sharpe": annual_sharpe,
    }

# %% [cell 14]
# Construct long-short portfolios for EW and VW
ew_ls_df = construct_long_short_portfolio(ew_portfolio_returns_df)
vw_ls_df = construct_long_short_portfolio(vw_portfolio_returns_df)
rw_ls_df = construct_long_short_portfolio(rw_portfolio_returns_df)

# Calculate statistics
ew_ls_stats = calculate_ls_statistics(ew_ls_df, ff_factors_df)
vw_ls_stats = calculate_ls_statistics(vw_ls_df, ff_factors_df)
rw_ls_stats = calculate_ls_statistics(rw_ls_df, ff_factors_df)

# Display results
results_df = pd.DataFrame(
    {
        "Equal-Weighted": ew_ls_stats,
        "Value-Weighted": vw_ls_stats,
        "Rank-Weighted": rw_ls_stats,
    }
).T

print("Long-Short Portfolio (Decile 10 - Decile 1) Statistics:")
display(results_df.round(4))

# %% [cell 15]
def construct_tau_horizon_ls_portfolio(
    signal_df: pl.DataFrame,
    signal_name: str,
    return_df: pl.DataFrame,
    tau: int,
    weighting: str = "ew",
) -> pl.DataFrame:
    """
    Construct a tau-month horizon long-short momentum portfolio.
    At month t, stocks are sorted into deciles based on month-t momentum signal.
    At month t+tau, go long top decile and short bottom decile.

    Args:
        signal_df: DataFrame with signal (permno, date, signal_name)
        signal_name: Name of the signal column
        return_df: DataFrame with returns (must have ret_p{tau} column)
        tau: Holding period in months
        weighting: "ew" for equal-weighted, "vw" for value-weighted
    """
    ret_col = f"ret_p{tau}"

    # Join signal with forward returns at horizon tau
    if weighting == "ew":
        df = signal_df.join(
            return_df.select("permno", "date", ret_col),
            on=["permno", "date"],
            how="left",
            validate="1:1",
        )
    else:  # vw
        df = signal_df.join(
            return_df.select("permno", "date", ret_col, "market_cap"),
            on=["permno", "date"],
            how="left",
            validate="1:1",
        )

    # Rank signals within each date and assign to deciles
    df = df.with_columns(
        pl.col(signal_name).rank(method="average").over("date").alias("rank")
    )

    # Calculate decile assignments (1-10)
    df = df.with_columns(
        ((pl.col("rank") - 1) / pl.col("rank").count().over("date") * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .add(1)
        .alias("decile")
    )

    # Calculate portfolio returns for each decile
    if weighting == "ew":
        portfolio_returns = (
            df.group_by(["date", "decile"])
            .agg(pl.col(ret_col).mean().alias("portfolio_return"))
            .sort(["date", "decile"])
        )
    else:  # vw
        portfolio_returns = (
            df.group_by(["date", "decile"])
            .agg(
                (pl.col(ret_col) * (pl.col("market_cap") / pl.col("market_cap").sum()))
                .sum()
                .alias("portfolio_return")
            )
            .sort(["date", "decile"])
        )

    # Pivot and compute long-short
    pivoted = portfolio_returns.pivot(
        index="date",
        on="decile",
        values="portfolio_return",
        aggregate_function=None,
    )

    ls_df = pivoted.select(
        pl.col("date"),
        (pl.col("10") - pl.col("1")).alias("ls_return"),
    )

    return ls_df


def calculate_ff_alpha_for_horizon(
    ls_df: pl.DataFrame, ff_factors_df: pl.DataFrame
) -> tuple[float, float]:
    """
    Calculate FF 3-factor alpha and t-statistic for a long-short portfolio.
    Returns (alpha, t_stat).
    """
    df = ls_df.join(
        ff_factors_df.select("date", "Mkt-RF", "SMB", "HML", "RF"),
        on="date",
        how="inner",
    )

    df = df.with_columns(
        (pl.col("ls_return") * 100 - pl.col("RF")).alias("excess_return")
    )

    df_pd = df.to_pandas().dropna()

    if len(df_pd) < 10:
        return np.nan, np.nan

    y = df_pd["excess_return"].values
    X = sm.add_constant(df_pd[["Mkt-RF", "SMB", "HML"]].values)
    model = sm.OLS(y, X, missing="drop").fit()

    return model.params[0], model.tvalues[0]

# %% [cell 16]
# Calculate FF alpha and t-stat for different horizons tau = 1 to 36
horizons = list(range(1, 37))
ew_results = []
vw_results = []

for tau in horizons:
    # Equal-weighted
    ew_ls = construct_tau_horizon_ls_portfolio(
        mom_12_1_df, "mom_12_1", ret_df, tau, weighting="ew"
    )
    ew_alpha, ew_tstat = calculate_ff_alpha_for_horizon(ew_ls, ff_factors_df)
    ew_results.append({"tau": tau, "ff_alpha": ew_alpha, "t_stat": ew_tstat})

    # Value-weighted
    vw_ls = construct_tau_horizon_ls_portfolio(
        mom_12_1_df, "mom_12_1", ret_df, tau, weighting="vw"
    )
    vw_alpha, vw_tstat = calculate_ff_alpha_for_horizon(vw_ls, ff_factors_df)
    vw_results.append({"tau": tau, "ff_alpha": vw_alpha, "t_stat": vw_tstat})

ew_horizon_df = pd.DataFrame(ew_results)
vw_horizon_df = pd.DataFrame(vw_results)

# %% [cell 17]
# Plot FF alpha and t-statistics as a function of horizon tau
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Equal-weighted FF alpha
axes[0, 0].bar(
    ew_horizon_df["tau"], ew_horizon_df["ff_alpha"], color="steelblue", alpha=0.7
)
axes[0, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
axes[0, 0].set_xlabel("Horizon tau (months)")
axes[0, 0].set_ylabel("FF Alpha (%)")
axes[0, 0].set_title("Equal-Weighted: FF Alpha by Horizon")

# Equal-weighted t-stat
axes[0, 1].bar(
    ew_horizon_df["tau"], ew_horizon_df["t_stat"], color="steelblue", alpha=0.7
)
axes[0, 1].axhline(y=1.96, color="red", linestyle="--", linewidth=0.8, label="t=1.96")
axes[0, 1].axhline(y=-1.96, color="red", linestyle="--", linewidth=0.8)
axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
axes[0, 1].set_xlabel("Horizon tau (months)")
axes[0, 1].set_ylabel("t-statistic")
axes[0, 1].set_title("Equal-Weighted: t-stat on FF Alpha by Horizon")
axes[0, 1].legend()

# Value-weighted FF alpha
axes[1, 0].bar(
    vw_horizon_df["tau"], vw_horizon_df["ff_alpha"], color="darkorange", alpha=0.7
)
axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
axes[1, 0].set_xlabel("Horizon tau (months)")
axes[1, 0].set_ylabel("FF Alpha (%)")
axes[1, 0].set_title("Value-Weighted: FF Alpha by Horizon")

# Value-weighted t-stat
axes[1, 1].bar(
    vw_horizon_df["tau"], vw_horizon_df["t_stat"], color="darkorange", alpha=0.7
)
axes[1, 1].axhline(y=1.96, color="red", linestyle="--", linewidth=0.8, label="t=1.96")
axes[1, 1].axhline(y=-1.96, color="red", linestyle="--", linewidth=0.8)
axes[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
axes[1, 1].set_xlabel("Horizon tau (months)")
axes[1, 1].set_ylabel("t-statistic")
axes[1, 1].set_title("Value-Weighted: t-stat on FF Alpha by Horizon")
axes[1, 1].legend()

plt.tight_layout()
plt.suptitle(
    "tau-Month Horizon Momentum Strategy: FF Alpha and t-statistics", y=1.02, fontsize=14
)
plt.show()
