import logging

from joblib import load
from sklearn.model_selection import train_test_split

from data import load_data, process_data
from model import compute_model_metrics


def compute_score(data_path, cat_features):
    df = load_data(data_path)
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    slice_values = []

    for cat in cat_features:
        for cl in test[cat].unique():
            df_temp = test[test[cat] == cl]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False,
            )

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " "Recall: %s FBeta: %s" % (
                cat,
                cl,
                prc,
                rcl,
                fb,
            )
            logging.info(line)
            slice_values.append(line)

    with open("model/slice_output.txt", "w") as out:
        for slice_value in slice_values:
            out.write(slice_value + "\n")


if __name__ == "__main__":
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

    compute_score("data/census.csv", cat_features)
