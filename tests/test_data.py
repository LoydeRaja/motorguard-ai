import pandas as pd

from src.motorguard_ai.data import prepare_training_data


def test_prepare_training_data_returns_expected_shapes():
    df = pd.DataFrame(
        {
            "individual_id": [1, 2],
            "address_id": [11, 12],
            "curr_ann_amt": [100.0, 200.0],
            "days_tenure": [10.0, 20.0],
            "cust_orig_date": ["2020-01-01", "2021-01-01"],
            "age_in_years": [30, 40],
            "date_of_birth": ["1990-01-01", "1980-01-01"],
            "latitude": [32.7, 32.8],
            "longitude": [-96.8, -96.9],
            "city": ["Dallas", "Plano"],
            "state": ["TX", "TX"],
            "county": ["Dallas", "Collin"],
            "income": [50000.0, 70000.0],
            "has_children": [1.0, 0.0],
            "length_of_residence": [5.0, 10.0],
            "marital_status": ["Married", "Single"],
            "home_market_value": ["50000 - 74999", "75000 - 99999"],
            "home_owner": [1.0, 1.0],
            "college_degree": [1.0, 0.0],
            "good_credit": [1.0, 0.0],
            "acct_suspd_date": [None, "2022-01-01"],
            "Churn": [0, 1],
        }
    )

    X, y = prepare_training_data(df)

    assert X.shape == (2, 16)
    assert y.shape == (2,)
    assert "Churn" not in X.columns
    assert "acct_suspd_date" not in X.columns
