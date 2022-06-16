from joblib import dump
from sklearn.model_selection import train_test_split

from src.data import load_data, process_data
from src.model import train_model

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

model = train_model(X_train, y_train)

dump(model, "model/model.joblib")
dump(encoder, "model/encoder.joblib")
dump(lb, "model/lb.joblib")
