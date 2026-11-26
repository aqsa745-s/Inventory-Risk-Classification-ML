from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "superstore_sample.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"

def demand_class(adi, cv2):
    if adi < 1.32 and cv2 < 0.49: return "Smooth"
    if adi < 1.32 and cv2 >= 0.49: return "Erratic"
    if adi >= 1.32 and cv2 < 0.49: return "Intermittent"
    return "Lumpy"

def risk_score(adi_percentile, cv2_percentile, sales_percentile):
    # Relative score within this dataset: intermittency 35%, variability 45%, sales exposure 20%.
    return round(35*adi_percentile + 45*cv2_percentile + 20*sales_percentile, 2)

def assign_risk(score):
    if score >= 67: return "High Risk"
    if score >= 34: return "Medium Risk"
    return "Low Risk"

def load_data(path=DATA):
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, encoding="latin1")
    required = ["Order Date", "Product ID", "Product Name", "Category",
                "Sub-Category", "Quantity", "Sales"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    for col in ["Quantity", "Sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    # Returned units are not treated as customer demand.
    df = df[df["Quantity"] > 0].copy()
    df["Month"] = df["Order Date"].dt.to_period("M")
    return df

def build_product_table(df):
    months = pd.period_range(df.Month.min(), df.Month.max(), freq="M")
    monthly = (df.groupby(["Product ID", "Month"], as_index=False)
                 .agg(demand=("Quantity", "sum"), sales=("Sales", "sum")))
    names = df.groupby("Product ID")["Product Name"].first()
    rows = []
    for pid, group in monthly.groupby("Product ID"):
        series = group.set_index("Month")["demand"].reindex(months, fill_value=0).astype(float)
        nonzero = series[series > 0]
        count = len(nonzero)
        adi = len(series) / count if count else float(len(series))
        nz_mean = float(nonzero.mean()) if count else 0.0
        nz_cv2 = (float(nonzero.std(ddof=0)) / nz_mean) ** 2 if nz_mean else 0.0
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        rows.append({
            "Product ID": pid, "Product Name": names.loc[pid],
            "Category": group.merge(df[["Product ID","Category"]].drop_duplicates(),
                                    on="Product ID", how="left")["Category"].iloc[0],
            "months_observed": len(series), "months_with_demand": count,
            "ADI": round(adi, 4), "CV2": round(nz_cv2, 4),
            "demand_pattern": demand_class(adi, nz_cv2),
            "avg_monthly_demand": round(mean, 4),
            "demand_std": round(std, 4),
            "total_sales": round(float(group.sales.sum()), 2),
            "total_units": round(float(group.demand.sum()), 2)
        })
    features = pd.DataFrame(rows)
    total_sales = features["total_sales"].sum()
    features["sales_share"] = features["total_sales"] / total_sales if total_sales else 0
    # Percentile ranks make the screening score comparable across products in this dataset.
    adi_rank = features["ADI"].rank(pct=True)
    cv2_rank = features["CV2"].rank(pct=True)
    sales_rank = features["sales_share"].rank(pct=True)
    features["risk_score"] = [
        risk_score(a, c, v) for a, c, v in zip(adi_rank, cv2_rank, sales_rank)
    ]
    features["risk_level"] = features["risk_score"].apply(assign_risk)
    features["cluster"] = -1
    # K-Means is used as an unsupervised grouping aid; risk labels remain interpretable.
    X = np.log1p(features[["ADI", "CV2"]].clip(lower=0))
    if len(features) >= 3:
        scaled = StandardScaler().fit_transform(X)
        k = min(3, len(features)-1)
        km = KMeans(n_clusters=k, random_state=27, n_init=15)
        labels = km.fit_predict(scaled)
        features["cluster"] = labels
        if len(set(labels)) > 1:
            score = silhouette_score(scaled, labels)
        else:
            score = np.nan
    else:
        score = np.nan
    return features, monthly, months, score

def build_abc_xyz(df, level):
    monthly = (df.groupby([level, "Month"], as_index=False)
                 .agg(demand=("Quantity","sum"), sales=("Sales","sum")))
    all_months = pd.period_range(df.Month.min(), df.Month.max(), freq="M")
    rows=[]
    for key, g in monthly.groupby(level):
        s=g.set_index("Month")["demand"].reindex(all_months,fill_value=0).astype(float)
        mean=s.mean()
        cv=s.std(ddof=0)/mean if mean else 0
        rows.append({level:key,"total_sales":g.sales.sum(),"mean_monthly_demand":mean,"demand_cv":cv})
    result=pd.DataFrame(rows).sort_values("total_sales",ascending=False).reset_index(drop=True)
    total=result.total_sales.sum()
    result["sales_share"]=result.total_sales/total if total else 0
    result["cumulative_share"]=result.sales_share.cumsum()
    result["ABC"]=np.select([result.cumulative_share<=.80,result.cumulative_share<=.95],["A","B"],default="C")
    result["XYZ"]=np.select([result.demand_cv<=.5,result.demand_cv<=1.0],["X","Y"],default="Z")
    result["ABC_XYZ"]=result.ABC+"-"+result.XYZ
    return result

def save_charts(features, monthly):
    CHARTS.mkdir(parents=True,exist_ok=True)
    counts=features["risk_level"].value_counts().reindex(["Low Risk","Medium Risk","High Risk"],fill_value=0)
    fig,ax=plt.subplots(figsize=(7,4)); counts.plot(kind="bar",ax=ax)
    ax.set_title("Products by Inventory Risk"); ax.set_xlabel("Risk level"); ax.set_ylabel("Number of products")
    fig.tight_layout(); fig.savefig(CHARTS/"risk_distribution.png",dpi=150); plt.close(fig)
    patterns=features["demand_pattern"].value_counts()
    fig,ax=plt.subplots(figsize=(7,4)); patterns.plot(kind="bar",ax=ax)
    ax.set_title("Demand Pattern Distribution"); ax.set_xlabel("Demand pattern"); ax.set_ylabel("Products")
    fig.tight_layout(); fig.savefig(CHARTS/"demand_patterns.png",dpi=150); plt.close(fig)
    totals=monthly.groupby("Month")["demand"].sum().sort_index()
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(totals.index.astype(str),totals.values,marker="o")
    ax.set_title("Total Monthly Demand"); ax.set_xlabel("Month"); ax.set_ylabel("Units")
    ax.tick_params(axis="x",rotation=70); fig.tight_layout()
    fig.savefig(CHARTS/"monthly_demand.png",dpi=150); plt.close(fig)

def run_analysis():
    OUT.mkdir(exist_ok=True); CHARTS.mkdir(exist_ok=True)
    df=load_data()
    features,monthly,months,silhouette=build_product_table(df)
    features.to_csv(OUT/"product_risk_report.csv",index=False)
    monthly.assign(Month=monthly.Month.astype(str)).to_csv(OUT/"monthly_product_demand.csv",index=False)
    build_abc_xyz(df,"Category").to_csv(OUT/"category_abc_xyz.csv",index=False)
    build_abc_xyz(df,"Sub-Category").to_csv(OUT/"subcategory_abc_xyz.csv",index=False)
    save_charts(features,monthly)
    summary=pd.DataFrame([
        ["Clean transactions",len(df)],["Unique products",features["Product ID"].nunique()],
        ["Monthly periods",len(months)],["High-risk products",(features.risk_level=="High Risk").sum()],
        ["Medium-risk products",(features.risk_level=="Medium Risk").sum()],
        ["Low-risk products",(features.risk_level=="Low Risk").sum()],
        ["K-Means silhouette score",silhouette]
    ],columns=["Metric","Value"])
    summary.to_csv(OUT/"project_summary.csv",index=False)
    print(summary.to_string(index=False))
    print(f"\nReports and charts saved in: {OUT}")
    return features,monthly

if __name__=="__main__":
    run_analysis()
