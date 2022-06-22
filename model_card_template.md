# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Mateus created the model. It is Decision Tree Classifier using the default hyperparameters in scikit-learn.

## Intended Use
The model should be used to predict a person's salary based on features about their finances.

## Training Data
The source of data is https://archive.ics.uci.edu/ml/datasets/census+income, 80% for training.

## Evaluation Data
The source of data is https://archive.ics.uci.edu/ml/datasets/census+income, 20% for evaluation.

## Metrics
Model accuracy score ~0.78.

## Ethical Considerations
The dataset contains data related race, gender and origin country. This will drive to a model that may potentially discriminate people.
