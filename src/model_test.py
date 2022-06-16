import numpy as np
import pandas as pd
import pytest
from joblib import load
from pandas.core.frame import DataFrame

from src.data import clean_data, process_data
from src.model import inference


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    df = clean_data(df)

    return df


@pytest.fixture
def categorical_features():
    """
    Set categorical features
    """
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

    return cat_features


def test_process_encoder(data, cat_features):
    """
    Check if split have same number of rows for X and y
    """
    encoder_test = load("model/encoder.joblib")
    lb_test = load("model/lb.joblib")

    _, _, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    _, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        encoder=encoder_test,
        lb=lb_test,
        training=False,
    )

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference_above(cat_features):
    """
    Test inference performance
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array(
        [
            [
                32,
                "Private",
                "Some-college",
                "Married-civ-spouse",
                "Exec-managerial",
                "Husband",
                "Black",
                "Male",
                80,
                "United-States",
            ]
        ]
    )
    df_temp = DataFrame(
        data=array,
        columns=[
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "hours-per-week",
            "native-country",
        ],
    )

    X, _, _, _ = process_data(
        df_temp,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False,
    )
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


def test_inference_below(cat_features):
    """
    Test inference performance
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array(
        [
            [
                19,
                "Private",
                "HS-grad",
                "Never-married",
                "Own-child",
                "Husband",
                "Black",
                "Male",
                40,
                "United-States",
            ]
        ]
    )
    df_temp = DataFrame(
        data=array,
        columns=[
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "hours-per-week",
            "native-country",
        ],
    )

    X, _, _, _ = process_data(
        df_temp,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False,
    )
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
