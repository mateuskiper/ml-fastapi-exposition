import numpy as np
from fastapi import FastAPI
from joblib import load
from pandas.core.frame import DataFrame
from src.data import process_data
from src.model import inference

from app.pydantic_classes import User

app = FastAPI()


@app.get("/health")
def health():
    return {"msg": "Up"}


@app.post("/predict")
def predict(user_data: User):
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array(
        [
            [
                user_data.age,
                user_data.workclass,
                user_data.education,
                user_data.maritalStatus,
                user_data.occupation,
                user_data.relationship,
                user_data.race,
                user_data.sex,
                user_data.hoursPerWeek,
                user_data.nativeCountry,
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
    X_pred, _, _, _ = process_data(
        df_temp,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False,
    )
    preds = inference(model, X_pred)
    y = lb.inverse_transform(preds)[0]

    return {"prediction": y}
