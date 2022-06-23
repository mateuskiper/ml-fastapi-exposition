import logging

from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data import load_data, process_data
from model import train_model
from utils import compute_score

data = load_data("data/census.csv")
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    encoder=encoder,
    lb=lb,
    training=False,
)

model = train_model(X_train, y_train)

y_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, y_preds)

logging.info("evaluation accuracy: %.2f" % test_acc)

dump(model, "model/model.joblib")
dump(encoder, "model/encoder.joblib")
dump(lb, "model/lb.joblib")

# Compute score
compute_score(test, cat_features, model, encoder, lb)
