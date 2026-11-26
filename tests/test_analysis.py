import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import demand_class, risk_score, assign_risk, load_data, build_product_table

def test_demand_classes():
    assert demand_class(1.0, 0.2) == "Smooth"
    assert demand_class(1.0, 0.8) == "Erratic"
    assert demand_class(2.0, 0.2) == "Intermittent"
    assert demand_class(2.0, 0.8) == "Lumpy"

def test_risk_score_bands():
    assert assign_risk(20) == "Low Risk"
    assert assign_risk(50) == "Medium Risk"
    assert assign_risk(80) == "High Risk"
    assert 0 <= risk_score(0.1, 0.2, 0.3) <= 100

def test_dataset_and_product_features():
    df = load_data()
    features, monthly, months, silhouette = build_product_table(df)
    assert not df.empty
    assert features["Product ID"].is_unique
    assert {"ADI", "CV2", "risk_score", "risk_level", "cluster"}.issubset(features.columns)
    assert set(features["risk_level"]).issubset({"Low Risk", "Medium Risk", "High Risk"})
