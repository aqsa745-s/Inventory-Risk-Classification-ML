from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from analysis import run_analysis, load_data, build_product_table, OUT

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Inventory Risk Intelligence",page_icon="📦",layout="wide")

@st.cache_data
def get_reports():
    features_file=OUT/"product_risk_report.csv"
    monthly_file=OUT/"monthly_product_demand.csv"
    if not features_file.exists() or not monthly_file.exists():
        run_analysis()
    return pd.read_csv(features_file),pd.read_csv(monthly_file)

def croston_sba(values,horizon,alpha=.1):
    x=np.asarray(values,dtype=float)
    positions=np.flatnonzero(x>0)
    if len(positions)==0: return np.zeros(horizon)
    demand=float(x[positions[0]])
    interval=float(positions[0]+1)
    previous=positions[0]
    for pos in positions[1:]:
        gap=float(pos-previous)
        demand=alpha*x[pos]+(1-alpha)*demand
        interval=alpha*gap+(1-alpha)*interval
        previous=pos
    estimate=(1-alpha/2)*demand/interval if interval else 0
    return np.repeat(max(0,estimate),horizon)

def forecast(series,horizon,method):
    vals=np.asarray(series,dtype=float)
    if method=="Moving average":
        estimate=float(np.mean(vals[-min(3,len(vals)):])) if len(vals) else 0
        return np.repeat(max(0,estimate),horizon),"Three-period moving-average baseline."
    if method=="Croston SBA":
        return croston_sba(vals,horizon),"Croston SBA estimate for intermittent demand."
    try:
        from statsmodels.tsa.arima.model import ARIMA
        if len(vals)<12: raise ValueError("At least 12 monthly observations are required.")
        model=ARIMA(vals,order=(1,1,1)).fit()
        return np.maximum(model.forecast(horizon),0),"ARIMA(1,1,1) fitted to the selected product history."
    except Exception as exc:
        ma=float(np.mean(vals[-min(3,len(vals)):])) if len(vals) else 0
        return np.repeat(max(0,ma),horizon),f"ARIMA could not be fitted ({exc}); moving-average fallback shown."

st.title("📦 Inventory Risk Classification")
st.caption("Demand analytics • Machine-learning segmentation • Inventory planning")
st.markdown("This decision-support dashboard identifies products with potentially unstable or intermittent demand. Risk levels are transparent screening indicators, not a substitute for live stock and supplier data.")

try:
    features,monthly=get_reports()
except Exception as e:
    st.error(f"Could not load project data: {e}")
    st.stop()

with st.sidebar:
    st.header("Dashboard controls")
    st.write("Select a product and planning assumptions.")
    product_ids=features["Product ID"].astype(str).tolist()
    selected=st.selectbox("Product ID",product_ids)
    lead=st.number_input("Supplier lead time (months)",min_value=.25,max_value=12.0,value=1.0,step=.25)
    service=st.selectbox("Target service level",[.90,.95,.975,.99],index=1)
    stock=st.number_input("Current stock (units)",min_value=0.0,value=20.0,step=1.0)
    horizon=st.slider("Forecast horizon (months)",1,12,6)
    method=st.selectbox("Forecast method",["Moving average","Croston SBA","ARIMA (1,1,1)"])

total=len(features)
high=int((features.risk_level=="High Risk").sum())
medium=int((features.risk_level=="Medium Risk").sum())
low=int((features.risk_level=="Low Risk").sum())
c1,c2,c3,c4=st.columns(4)
c1.metric("Products analysed",f"{total:,}")
c2.metric("High risk",f"{high:,}")
c3.metric("Medium risk",f"{medium:,}")
c4.metric("Low risk",f"{low:,}")

st.subheader("Inventory risk overview")
left,right=st.columns([1,1])
with left:
    risk_counts=features.risk_level.value_counts().reindex(["Low Risk","Medium Risk","High Risk"],fill_value=0)
    fig,ax=plt.subplots(figsize=(6,4)); risk_counts.plot(kind="bar",ax=ax)
    ax.set_ylabel("Products"); ax.set_xlabel("Risk level"); ax.set_title("Risk classification")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
with right:
    pattern_counts=features.demand_pattern.value_counts()
    fig,ax=plt.subplots(figsize=(6,4)); pattern_counts.plot(kind="bar",ax=ax)
    ax.set_ylabel("Products"); ax.set_xlabel("Demand pattern"); ax.set_title("Demand pattern profile")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.subheader("Product risk register")
risk_filter=st.multiselect("Filter risk levels",["Low Risk","Medium Risk","High Risk"],default=["Low Risk","Medium Risk","High Risk"])
filtered=features[features.risk_level.isin(risk_filter)].sort_values("risk_score",ascending=False)
st.dataframe(filtered[["Product ID","Product Name","Category","demand_pattern","ADI","CV2","risk_score","risk_level","total_sales","cluster"]],use_container_width=True)
st.download_button("Download risk report (CSV)",filtered.to_csv(index=False).encode("utf-8"),"inventory_risk_report.csv","text/csv")

row=features.loc[features["Product ID"].astype(str)==selected].iloc[0]
history=monthly.loc[monthly["Product ID"].astype(str)==selected].copy()
history["Month"]=pd.PeriodIndex(history["Month"],freq="M")
calendar=pd.period_range(monthly["Month"].min(),monthly["Month"].max(),freq="M")
series=history.set_index("Month")["demand"].reindex(calendar,fill_value=0).astype(float)
zmap={.90:1.282,.95:1.645,.975:1.960,.99:2.326}
z=zmap[service]
safety=z*float(series.std(ddof=0))*np.sqrt(lead)
rop=float(series.mean())*lead+safety
risk_color={"Low Risk":"🟢","Medium Risk":"🟠","High Risk":"🔴"}.get(row.risk_level,"")
st.divider()
st.subheader(f"Selected product: {row['Product Name']}")
a,b,c,d=st.columns(4)
a.metric("Risk classification",f"{risk_color} {row.risk_level}")
b.metric("Risk score",f"{row.risk_score:.1f}/100")
c.metric("Demand pattern",row.demand_pattern)
d.metric("K-Means group",str(int(row.cluster)))

st.subheader("Inventory planning scenario")
a,b,c=st.columns(3)
a.metric("Estimated safety stock",f"{safety:.1f} units")
b.metric("Estimated reorder point",f"{rop:.1f} units")
c.metric("Stock position", "Review reorder" if stock<=rop else "Above estimated ROP")
st.caption(f"Scenario assumptions: {lead:g}-month lead time and {service:.1%} service level. The supplied sample dataset does not contain actual on-hand stock or supplier lead-time records.")

st.subheader("Historical demand and forecast")
forecast_method=method
if forecast_method=="ARIMA (1,1,1)": forecast_method="ARIMA"
pred,note=forecast(series.values,horizon,forecast_method)
future=pd.period_range(calendar[-1]+1,periods=horizon,freq="M")
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(series.index.astype(str),series.values,label="Historical demand",marker="o")
ax.plot(future.astype(str),pred,label="Forecast",marker="o",linestyle="--")
ax.set_xlabel("Month"); ax.set_ylabel("Units"); ax.tick_params(axis="x",rotation=70); ax.legend()
fig.tight_layout(); st.pyplot(fig); plt.close(fig)
st.info(note)

tab1,tab2,tab3=st.tabs(["ABC-XYZ analysis","Methodology","Project limitations"])
with tab1:
    for label in ["category_abc_xyz.csv","subcategory_abc_xyz.csv"]:
        file=OUT/label
        if file.exists():
            table=pd.read_csv(file)
            st.markdown(f"**{label.replace('_',' ').replace('.csv','').title()}**")
            st.dataframe(table,use_container_width=True)
with tab2:
    st.markdown("""
    - **ADI:** observed calendar months divided by months with non-zero demand.
    - **CV²:** squared coefficient of variation of non-zero monthly demand.
    - **Demand patterns:** Smooth, Erratic, Intermittent and Lumpy, using ADI 1.32 and CV² 0.49 reference cut-offs.
    - **K-Means:** groups products using log-transformed ADI and CV². It is an unsupervised exploration tool.
    - **Risk score:** combines within-dataset percentile ranks: ADI (35 points), CV² (45 points) and sales exposure (20 points). Low is below 34, Medium is 34–66.99, High is 67 or above.
    - **Safety stock:** z × historical monthly demand standard deviation × square root of assumed lead time.
    - **Reorder point:** average monthly demand × lead time + safety stock.
    """)
with tab3:
    st.warning("""
    This is an academic prototype built on sample retail transactions. The data has no verified inventory-on-hand,
    supplier lead-time history, purchase orders or stockout labels. Risk levels are screening categories created
    from transparent heuristics, not supervised predictions trained on confirmed stockout outcomes. Forecasts and
    inventory quantities must be validated with real operational data before business use.
    """)
st.caption("Academic decision-support prototype • See README.md and SOURCES.md for data and methodology notes.")
