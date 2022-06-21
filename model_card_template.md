# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Mateus created the model. It is Decision Tree Classifier using the default hyperparameters in scikit-learn.

## Intended Use
The model should be used to predict a person's salary based on features about their finances.

## Data
Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income, training is done using 80% of this data and evaluation is done using 20%.

## Ethical Considerations
The dataset contains data related race, gender and origin country. This will drive to a model that may potentially discriminate people.
