from datetime import date
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "report_assets"
ASSETS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 6)


def read_fred_csv(series_id, start_date):
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}&observation_start={start_date:%Y-%m-%d}"
    )
    data = pd.read_csv(url)
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    data = data.rename(columns={"observation_date": "date", series_id: series_id.lower()})
    data[series_id.lower()] = pd.to_numeric(data[series_id.lower()], errors="coerce")
    return data.set_index("date")


def fit_ols(y_column, x_columns, data_frame):
    model_data = data_frame[[y_column] + x_columns].dropna()
    y = model_data[y_column]
    x = sm.add_constant(model_data[x_columns])
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


def fmt_number(value, decimals=2):
    return f"{value:.{decimals}f}"


def effect_text(value, decimals=3):
    direction = "lower" if value < 0 else "higher"
    return f"{abs(value):.{decimals}f} percentage points {direction}"


def latex_escape(value):
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_data():
    today = pd.Timestamp(date.today())
    analysis_start = (today - pd.DateOffset(years=10)).to_period("M").to_timestamp()
    download_start = analysis_start - pd.DateOffset(years=1)

    unrate = read_fred_csv("UNRATE", download_start)
    cpi = read_fred_csv("CPIAUCSL", download_start)
    core_cpi = read_fred_csv("CPILFESL", download_start)
    natural_rate = read_fred_csv("NROU", download_start)

    raw_data = unrate.join([cpi, core_cpi, natural_rate], how="outer").sort_index()
    raw_data["nrou"] = raw_data["nrou"].ffill()
    raw_data = raw_data.dropna(subset=["unrate", "cpiaucsl", "cpilfesl", "nrou"])

    data = raw_data.copy()
    data["inflation_yoy"] = data["cpiaucsl"].pct_change(12, fill_method=None) * 100
    data["core_inflation_yoy"] = data["cpilfesl"].pct_change(12, fill_method=None) * 100
    data = data.rename(columns={"unrate": "unemployment_rate"})
    data["unemployment_gap"] = data["unemployment_rate"] - data["nrou"]
    data["unemployment_gap_squared"] = data["unemployment_gap"] ** 2
    data["core_inflation_yoy_lag1"] = data["core_inflation_yoy"].shift(1)
    data["covid_dummy"] = ((data.index >= "2020-03-01") & (data.index <= "2021-12-01")).astype(int)

    columns = [
        "unemployment_rate",
        "nrou",
        "unemployment_gap",
        "unemployment_gap_squared",
        "inflation_yoy",
        "core_inflation_yoy",
        "core_inflation_yoy_lag1",
        "covid_dummy",
    ]
    monthly_data = data.loc[data.index >= analysis_start, columns].dropna()
    return monthly_data


def make_figures(monthly_data):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    monthly_data["unemployment_rate"].plot(ax=axes[0], color="#1f77b4", linewidth=2)
    axes[0].set_title("US Unemployment Rate")
    axes[0].set_ylabel("Percent")

    monthly_data["inflation_yoy"].plot(ax=axes[1], color="#d62728", linewidth=2)
    monthly_data["core_inflation_yoy"].plot(ax=axes[1], color="#9467bd", linewidth=2)
    axes[1].set_title("US Inflation")
    axes[1].set_ylabel("Year-over-year percent")
    axes[1].legend(["Headline CPI", "Core CPI"])
    plt.tight_layout()
    fig.savefig(ASSETS / "labor_inflation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.regplot(
        data=monthly_data,
        x="unemployment_gap",
        y="core_inflation_yoy",
        ax=axes[0],
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "black", "linewidth": 2},
    )
    axes[0].axvline(0, color="gray", linewidth=1, linestyle="--")
    axes[0].set_title("Linear Fit")
    axes[0].set_xlabel("Unemployment gap (p.p.)")
    axes[0].set_ylabel("Core CPI inflation (%)")

    sns.regplot(
        data=monthly_data,
        x="unemployment_gap",
        y="core_inflation_yoy",
        ax=axes[1],
        order=2,
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "black", "linewidth": 2},
    )
    axes[1].axvline(0, color="gray", linewidth=1, linestyle="--")
    axes[1].set_title("Quadratic Fit")
    axes[1].set_xlabel("Unemployment gap (p.p.)")
    axes[1].set_ylabel("Core CPI inflation (%)")
    plt.tight_layout()
    fig.savefig(ASSETS / "phillips_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def build_report(monthly_data):
    simple_model = fit_ols("inflation_yoy", ["unemployment_rate"], monthly_data)
    gap_model = fit_ols("inflation_yoy", ["unemployment_gap"], monthly_data)
    augmented_linear = fit_ols(
        "core_inflation_yoy",
        ["unemployment_gap", "core_inflation_yoy_lag1"],
        monthly_data,
    )
    augmented_quadratic = fit_ols(
        "core_inflation_yoy",
        ["unemployment_gap", "unemployment_gap_squared", "core_inflation_yoy_lag1"],
        monthly_data,
    )

    non_covid_data = monthly_data.loc[monthly_data["covid_dummy"] == 0].copy()
    augmented_ex_covid = fit_ols(
        "core_inflation_yoy",
        ["unemployment_gap", "unemployment_gap_squared", "core_inflation_yoy_lag1"],
        non_covid_data,
    )

    granger_data = monthly_data[["core_inflation_yoy", "unemployment_gap"]].dropna()
    max_lag = 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        gap_predicts_inflation = grangercausalitytests(
            granger_data[["core_inflation_yoy", "unemployment_gap"]],
            maxlag=max_lag,
            verbose=False,
        )
        inflation_predicts_gap = grangercausalitytests(
            granger_data[["unemployment_gap", "core_inflation_yoy"]],
            maxlag=max_lag,
            verbose=False,
        )
    best_gap_to_inflation_p = min(
        gap_predicts_inflation[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
    )
    best_inflation_to_gap_p = min(
        inflation_predicts_gap[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
    )

    latest = monthly_data.iloc[-1]
    latest_gap = latest["unemployment_gap"]
    sample_start = monthly_data.index.min().strftime("%B %Y")
    sample_end = monthly_data.index.max().strftime("%B %Y")
    latest_month = monthly_data.index.max().strftime("%B %Y")
    unemployment_max_date = monthly_data["unemployment_rate"].idxmax().strftime("%B %Y")
    inflation_max_date = monthly_data["inflation_yoy"].idxmax().strftime("%B %Y")
    report_month_year = date.today().strftime("%B %Y")

    model_rows = [
        [
            "Simple model",
            "Headline inflation",
            "Unemployment rate",
            fmt_number(simple_model.params["unemployment_rate"], 3),
            fmt_number(simple_model.pvalues["unemployment_rate"], 3),
            fmt_number(simple_model.rsquared, 3),
        ],
        [
            "Gap model",
            "Headline inflation",
            "Unemployment gap",
            fmt_number(gap_model.params["unemployment_gap"], 3),
            fmt_number(gap_model.pvalues["unemployment_gap"], 3),
            fmt_number(gap_model.rsquared, 3),
        ],
        [
            "Augmented linear",
            "Core inflation",
            "Unemployment gap",
            fmt_number(augmented_linear.params["unemployment_gap"], 3),
            fmt_number(augmented_linear.pvalues["unemployment_gap"], 3),
            fmt_number(augmented_linear.rsquared, 3),
        ],
        [
            "Augmented quadratic",
            "Core inflation",
            "Gap + gap squared",
            fmt_number(augmented_quadratic.params["unemployment_gap"], 3),
            fmt_number(augmented_quadratic.pvalues["unemployment_gap"], 3),
            fmt_number(augmented_quadratic.rsquared, 3),
        ],
    ]

    table_lines = []
    for row in model_rows:
        table_lines.append(" & ".join(latex_escape(item) for item in row) + r" \\")
    model_table = "\n".join(table_lines)

    gap_coef_linear = augmented_linear.params["unemployment_gap"]
    gap_pvalue_linear = augmented_linear.pvalues["unemployment_gap"]
    gap_coef_quad = augmented_quadratic.params["unemployment_gap"]
    gap_sq_coef = augmented_quadratic.params["unemployment_gap_squared"]
    gap_pvalue_quad = augmented_quadratic.pvalues["unemployment_gap"]
    gap_sq_pvalue = augmented_quadratic.pvalues["unemployment_gap_squared"]
    lag_coef = augmented_quadratic.params["core_inflation_yoy_lag1"]
    lag_pvalue = augmented_quadratic.pvalues["core_inflation_yoy_lag1"]
    marginal_effect_latest = gap_coef_quad + 2 * gap_sq_coef * latest_gap
    marginal_effect_normal = gap_coef_quad
    marginal_effect_latest_text = effect_text(marginal_effect_latest)
    marginal_effect_normal_text = effect_text(marginal_effect_normal)
    ex_covid_gap = augmented_ex_covid.params["unemployment_gap"]
    ex_covid_pvalue = augmented_ex_covid.pvalues["unemployment_gap"]

    report = rf"""
\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{hyperref}}
\geometry{{margin=1in}}

\title{{US Phillips Curve: Inflation and Labor Market Conditions}}
\author{{Macroeconomic Research Note}}
\date{{{report_month_year}}}

\begin{{document}}
\maketitle

\section*{{Executive Summary}}
This short report studies the recent relationship between inflation and unemployment in the United States. The sample covers monthly data from {sample_start} to {sample_end}. The main conclusion is simple: in the last decade, the traditional Phillips curve relationship is weak in the raw data. Inflation persistence is much stronger than the direct link between inflation and unemployment.

In {latest_month}, the unemployment rate was {fmt_number(latest["unemployment_rate"])}\%, headline CPI inflation was {fmt_number(latest["inflation_yoy"])}\%, and core CPI inflation was {fmt_number(latest["core_inflation_yoy"])}\%.

\section*{{Variables Used}}
\begin{{itemize}}
    \item \textbf{{Unemployment rate:}} the share of the labor force unemployed, from FRED series UNRATE.
    \item \textbf{{Headline CPI inflation:}} year-over-year percentage change in CPI, from FRED series CPIAUCSL.
    \item \textbf{{Core CPI inflation:}} year-over-year percentage change in CPI excluding food and energy, from FRED series CPILFESL.
    \item \textbf{{Natural rate of unemployment:}} an estimate of the unemployment rate consistent with normal labor-market conditions, from FRED series NROU.
    \item \textbf{{Unemployment gap:}} unemployment rate minus the natural rate. A negative gap means the labor market is tighter than normal.
\end{{itemize}}

\section*{{What Happened in Recent Years}}
The last decade includes three very different macroeconomic phases. Before 2020, unemployment was low and inflation was relatively contained. In 2020, the COVID shock created an abrupt labor-market deterioration, with unemployment peaking in {unemployment_max_date}. After that, unemployment recovered quickly, while inflation accelerated sharply and reached its sample peak in {inflation_max_date}.

This matters for the Phillips curve because the period was not a normal business cycle. Supply shocks, fiscal support, pandemic reopening, energy prices, and inflation expectations all affected inflation. For that reason, a simple inflation-versus-unemployment chart can be informative, but it is not enough to explain inflation alone.

\begin{{figure}}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{{report_assets/labor_inflation.pdf}}
    \caption{{Unemployment, headline inflation, and core inflation.}}
\end{{figure}}

\section*{{How Inflation and Unemployment Interacted}}
The traditional Phillips curve says that inflation should rise when unemployment is low, because a tight labor market can increase wage and price pressures. In practice, the relationship may be curved rather than a straight line. We therefore compare a linear fit with a quadratic fit, which allows the relationship to bend when the labor market is very tight or very weak.

\begin{{figure}}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{{report_assets/phillips_scatter.pdf}}
    \caption{{Phillips curve views: linear and quadratic fits using the unemployment gap.}}
\end{{figure}}

\section*{{Model Results}}
We estimated four simple models. The final specification uses core inflation, the unemployment gap, the square of the unemployment gap, and lagged core inflation. The squared term allows for a curved Phillips curve. Lagged inflation is included because inflation is usually persistent: this month's inflation is strongly related to recent inflation.

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{p{{2.8cm}}p{{3.0cm}}p{{3.1cm}}rrr}}
\toprule
Model & Dependent variable & Labor-market variable & Coef. & p-value & R$^2$ \\
\midrule
{model_table}
\bottomrule
\end{{tabular}}
\caption{{Phillips curve model comparison. Standard errors use a HAC correction.}}
\end{{table}}

In the augmented linear model, the unemployment-gap coefficient is {fmt_number(gap_coef_linear, 4)} with a p-value of {fmt_number(gap_pvalue_linear, 4)}. This means the estimated relationship is negative, as the traditional Phillips curve would suggest, but it is not statistically strong in this recent 10-year sample.

The quadratic model changes the interpretation. The coefficient on the unemployment gap is {fmt_number(gap_coef_quad, 4)}, and the coefficient on the squared unemployment gap is {fmt_number(gap_sq_coef, 4)}. The squared term has a p-value of {fmt_number(gap_sq_pvalue, 4)}. At the latest unemployment gap, the model-implied association is that a 1 percentage point higher unemployment gap is linked to about {marginal_effect_latest_text} core inflation, holding lagged inflation constant. Around a normal labor market, where the unemployment gap is zero, the model-implied association is about {marginal_effect_normal_text}. This shows why the quadratic form should be read carefully: the implied effect is not constant across the cycle.

The lagged core inflation coefficient is {fmt_number(lag_coef, 4)} with a p-value of {fmt_number(lag_pvalue, 4)}. This indicates strong inflation persistence. In practical terms, recent inflation explains current inflation better than unemployment does in this sample.

When the COVID period is excluded, the unemployment-gap coefficient is {fmt_number(ex_covid_gap, 4)} with a p-value of {fmt_number(ex_covid_pvalue, 4)}. This check helps show whether the pandemic period is driving the conclusion.

\section*{{Causality and Practical Meaning}}
These regressions should be read as associations, not proof of causality. For example, the model can say that a 1 percentage point higher unemployment gap is associated with a certain change in inflation, holding other included variables constant. This is different from saying that unemployment mechanically caused inflation to fall by that amount. Also note the unit: unemployment rates move in percentage points, so a move from 4\% to 5\% unemployment is a 1 percentage point increase.

We also ran a Granger-causality check with up to six monthly lags. This test asks whether past unemployment gaps help predict current core inflation, and whether past core inflation helps predict the unemployment gap. The best p-value for unemployment helping predict inflation was {fmt_number(best_gap_to_inflation_p, 4)}. The best p-value for inflation helping predict unemployment was {fmt_number(best_inflation_to_gap_p, 4)}. Since both are above 0.05, the test does not provide strong evidence of predictive causality in either direction. This is a forecasting test, not proof of economic causality.

\section*{{Client-Friendly Interpretation}}
The recent US Phillips curve is not absent, but it is weak. Allowing the curve to bend adds flexibility, but it does not overturn the main conclusion: low unemployment alone did not explain the inflation surge of 2021--2023. A more complete explanation needs inflation persistence, pandemic disruptions, supply conditions, energy prices, and expectations.

For investors and bank clients, the main message is that inflation risk should not be judged only by the unemployment rate. Labor-market tightness matters, but recent inflation dynamics show that price pressures can remain elevated even when the direct unemployment-inflation relationship is statistically modest.

\section*{{Data Source}}
Data are from the Federal Reserve Bank of St. Louis FRED database: UNRATE, CPIAUCSL, CPILFESL, and NROU. Inflation rates are calculated as year-over-year percentage changes.

\end{{document}}
"""

    (ROOT / "phillips_curve_report.tex").write_text(report.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    monthly = build_data()
    make_figures(monthly)
    build_report(monthly)
    print("Report assets and LaTeX file generated.")
