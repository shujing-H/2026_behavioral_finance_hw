"""
Auto-generated from hw3/code/hw3.ipynb.

This file is intended for the homework report appendix.
"""

# flake8: noqa

# %% [cell 1]
import polars as pl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_1samp

# %% [cell 2]
data_df = pl.scan_csv(
    "../data/payments.csv",
).collect()
data_df.head()

# %% [cell 3]
# format the date
data_df = data_df.with_columns(
    pl.col("date")
    .str.strip_chars()
    .str.to_date(format="%d%b%Y", strict=False)
    .alias("date")
)
data_df.head()

# %% [cell 4]
def construct_abnormal_dividend_yield(
    data_df: pl.DataFrame, div_col: str
) -> pl.DataFrame:
    return (
        data_df.sort("date")
        .with_columns(
            pl.col(div_col).rolling_sum(window_size=2).alias(f"{div_col}_2"),
            pl.col(div_col)
            .rolling_mean(window_size=233, min_samples=233)
            .shift(20)
            .alias(f"avg_{div_col}"),
        )
        .with_columns(
            (pl.col(f"{div_col}_2") / pl.col(f"avg_{div_col}")).alias(
                f"abnormal_dividend_yield({div_col})"
            )
        )
    )

# %% [cell 5]
data_df2 = construct_abnormal_dividend_yield(data_df, "div_tot_paid")

# %% [cell 6]
# summary statistics
data_df2["abnormal_dividend_yield(div_tot_paid)"].describe()

# %% [cell 7]
regression_data = data_df2[
    ["vwretd", "ewretd", "abnormal_dividend_yield(div_tot_paid)"]
].drop_nulls()
X = sm.add_constant(
    regression_data[["abnormal_dividend_yield(div_tot_paid)"]].to_numpy()
)

# %% [cell 8]
# regression 1: vwretd ~ abnormal_dividend_yield
y = regression_data["vwretd"].to_numpy()
model1 = sm.OLS(y, X).fit(missing="drop")
print(model1.summary())

# %% [cell 9]
model1.rsquared

# %% [cell 10]
# model 2: ewretd ~ abnormal_dividend_yield
y = regression_data["ewretd"].to_numpy()
model2 = sm.OLS(y, X).fit(missing="drop")
print(model2.summary())

# %% [cell 11]
model2.rsquared

# %% [cell 12]
def construct_quintile_analysis(data_df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    quintile_df = data_df.drop_nulls(subset=["vwretd", group_col]).with_columns(
        pl.col(group_col)
        .qcut(5, labels=["q1", "q2", "q3", "q4", "q5"])
        .alias("quintile"),
    )

    results = []
    for quintile in ["q1", "q2", "q3", "q4", "q5"]:
        returns = quintile_df.filter(pl.col("quintile") == quintile)[
            "vwretd"
        ].to_numpy()

        mean_ret = returns.mean()
        mean_ady = (
            quintile_df.filter(pl.col("quintile") == quintile)[group_col]
            .to_numpy()
            .mean()
        )
        n_obs = len(returns)

        # Perform one-sample t-test (H0: mean = 0)
        t_stat, p_value = ttest_1samp(returns, 0)

        results.append(
            {
                "quintile": quintile,
                "mean_vwretd": mean_ret,
                "mean_abnormal_dividend_yield": mean_ady,
                "n_obs": n_obs,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    return pl.DataFrame(results)

# %% [cell 13]
construct_quintile_analysis(data_df2, "abnormal_dividend_yield(div_tot_paid)")

# %% [cell 14]
data_df3 = construct_abnormal_dividend_yield(data_df, "div_tot_recorded")
construct_quintile_analysis(data_df3, "abnormal_dividend_yield(div_tot_recorded)")
