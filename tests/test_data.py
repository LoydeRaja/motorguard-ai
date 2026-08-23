import pandas as pd
from src.data import clean_data

def test_clean_data_keeps_binary_target():
    df = pd.DataFrame({
        "curr_ann_amt": [100, 200, 300],
        "days_tenure": [10, 20, 30],
        "age_in_years": [30, 40, 50],
        "latitude": [1, None, 3],
        "longitude": [2, 3, None],
        "city": ["A", "B", None],
        "county": ["C1", "C2", "C3"],
        "income": [50000, 60000, 70000],
        "has_children": [0, 1, 0],
        "length_of_residence": [2, 5, 7],
        "marital_status": ["Single", "Married", "Single"],
        "home_market_value": ["50000 - 74999"] * 3,
        "home_owner": [1, 0, 1],
        "college_degree": [1, 1, 0],
        "good_credit": [1, 0, 1],
        "Churn": [0, 1, 0],
    })
    result = clean_data(df)
    assert len(result) == 3
    assert set(result["Churn"]) == {0, 1}
