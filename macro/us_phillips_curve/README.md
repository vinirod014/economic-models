# Macroeconomics

Applied macroeconomics projects using public data and reproducible Python workflows.

## Project 1: US Phillips Curve

This project studies the relationship between unemployment and inflation in the United States using monthly data from FRED.

The analysis includes:

- data collection from public FRED CSV endpoints;
- headline CPI inflation and core CPI inflation;
- unemployment rate and unemployment gap;
- stationarity tests;
- linear and quadratic Phillips curve specifications;
- Granger predictive-causality tests;
- a simple stakeholder-oriented PDF report.

## Main Files

- `phillips_curve.ipynb`: main notebook with explanations, charts, tests, and model results.
- `build_report.py`: script that downloads the data again, recreates report figures, and writes the LaTeX report.
- `phillips_curve_report.tex`: LaTeX source for the report.
- `phillips_curve_report.pdf`: latest compiled PDF report.
- `report_assets/`: figures used in the report.
- `requirements.txt`: Python packages needed to run the project.

## Data Source

Data are collected from the Federal Reserve Bank of St. Louis FRED database:

- `UNRATE`: US unemployment rate;
- `CPIAUCSL`: headline CPI index;
- `CPILFESL`: core CPI index;
- `NROU`: natural rate of unemployment.

Inflation is calculated as year-over-year percentage change.

## How To Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

Open the notebook:

```powershell
jupyter notebook phillips_curve.ipynb
```

Regenerate the report assets and LaTeX file:

```powershell
python build_report.py
```

Compile the PDF report if LaTeX is installed:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error phillips_curve_report.tex
pdflatex -interaction=nonstopmode -halt-on-error phillips_curve_report.tex
```

## Reproducibility Notes

This project does not require a FRED API key because it uses public CSV links.

Results may change slightly over time because FRED data can be revised and new monthly observations are added.

No private data, local credentials, or API keys are needed to run the project.

## Main Finding

For the recent US sample, inflation persistence is stronger than the direct unemployment-inflation relationship. The quadratic Phillips curve adds flexibility, but the results should be interpreted as statistical associations and predictive evidence, not proof of economic causality.
