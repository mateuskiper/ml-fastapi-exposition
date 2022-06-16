import pandas as pd
import pytest
from joblib import load
from src.data import clean_data, process_data


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    df = clean_data(df)

    return df


def test_null(data):
    """
    Data is assumed to have no null values
    """
    assert data.shape == data.dropna().shape


def test_question_mark(data):
    """
    Data is assumed to have no question marks value
    """
    assert "?" not in data.values


def test_removed_columns(data):
    """
    Data is assumed to have no question marks value
    """
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns


def test_process_data(data):
    """
    Check if split have same number of rows for X and y
    """
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False,
    )

    assert len(X_test) == len(y_test)
