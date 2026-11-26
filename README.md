# Inventory Risk Classification using Machine Learning

An independently implemented academic decision-support prototype for profiling retail product demand, exploring product groups with K-Means, assigning transparent inventory-risk screening levels, and estimating basic replenishment quantities.

## What the application does

- Cleans retail transaction data and removes returned/negative-quantity rows from demand calculations.
- Aggregates monthly demand for each Product ID.
- Calculates Average Demand Interval (ADI) and squared coefficient of variation (CV²).
- Classifies demand as Smooth, Erratic, Intermittent, or Lumpy.
- Uses K-Means clustering on log-transformed ADI and CV² as an exploratory unsupervised learning step.
- Calculates a transparent 0–100 inventory risk score and assigns Low, Medium, or High Risk.
- Produces ABC-XYZ category and sub-category summaries.
- Demonstrates Moving Average, Croston SBA, and ARIMA forecasting.
- Estimates Safety Stock and Reorder Point from editable scenario assumptions.
- Provides a Streamlit dashboard and downloadable CSV report.

## Important distinction: risk score vs. machine learning

The three risk levels in this prototype are produced by a documented scoring rule, not by a supervised classifier trained on historical stockout labels. K-Means is the machine-learning component: it groups products by demand interval and variability without pre-existing labels. The dashboard displays both outputs separately so that the method is not misrepresented.

Risk score components use each feature's percentile rank within the current dataset:
- ADI percentile: 35 points.
- CV² percentile: 45 points.
- Sales exposure percentile: 20 points.

Risk bands:
- Low Risk: below 34.
- Medium Risk: 34 to below 67.
- High Risk: 67 or above.

These are project-defined relative screening bands, not calibrated stockout probabilities. They should be recalibrated with a retailer's actual stockout, lead-time, and inventory records.

## Dataset

The included `data/superstore_sample.csv` is the Tableau Sample - Superstore sample transaction dataset. It is used for educational demonstration only. See [SOURCES.md](SOURCES.md). It is not live company inventory data and does not contain verified on-hand stock or supplier lead times.

## Run on Windows with VS Code

1. Extract the project ZIP.
2. Open the extracted `Inventory_Risk_Classification_ML` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

6. Generate analysis reports:

```powershell
python analysis.py
```

7. Launch the dashboard:

```powershell
streamlit run app.py
```

8. Open the local URL shown in the terminal (normally `http://localhost:8501`).

If PowerShell blocks environment activation, run:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe analysis.py
venv\Scripts\streamlit.exe run app.py
```

## Project structure

```text
Inventory_Risk_Classification_ML/
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
├── SOURCES.md
├── data/
│   └── superstore_sample.csv
├── tests/
│   └── test_analysis.py
└── outputs/                 # generated after running analysis.py
    ├── product_risk_report.csv
    ├── monthly_product_demand.csv
    ├── category_abc_xyz.csv
    ├── subcategory_abc_xyz.csv
    ├── project_summary.csv
    └── charts/
```

## Outputs

- `product_risk_report.csv`: product-level demand and risk indicators.
- `monthly_product_demand.csv`: monthly demand series by product.
- `category_abc_xyz.csv` and `subcategory_abc_xyz.csv`: ABC-XYZ summaries.
- `project_summary.csv`: high-level analysis counts.
- `outputs/charts/`: generated charts.

## Methodology and limitations

- ADI and CV² are descriptive demand-pattern measures.
- K-Means cluster labels are arbitrary group IDs; they are not risk labels.
- The risk score is a transparent heuristic. It is not a validated probability of stockout.
- The sample does not include actual inventory balances, stockout events, replenishment orders, or supplier lead times.
- Safety stock and reorder point are scenario estimates using an assumed lead time and normal-demand approximation.
- Forecast accuracy should be evaluated using a chronological holdout (for example MAE, RMSE, and WAPE) before operational use.

## Testing

Run:

```powershell
pytest -q
```
