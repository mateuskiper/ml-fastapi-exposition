import logging

from joblib import load

from data import process_data
from model import compute_model_metrics


def compute_score(test_data, cat_features, model, encoder, lb):
    slice_values = []

    for cat in cat_features:
        for cl in test_data[cat].unique():
            df_temp = test_data[test_data[cat] == cl]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False,
            )

            y_preds = model.predict(X_test)

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
