# Deploying a Machine Learning Model on Heroku with FastAPI

- Project **Deploying a Machine Learning Model on Heroku with FastAPI** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, a classification model was developed on publicly available data from the Census Bureau. To monitor the performance of the model on various slices of the data, unit tests were created. Then the model was deployed using the FastAPI package and the slice validation and API tests were embedded in a CI/CD framework using GitHub Actions.

## Running Project
1. Create a python environment
```bash
python3 -m venv venv
```

2. Activate the environment
```bash
source venv/bin/activate
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run tests
```bash
pytest
```

5. Data cleaning and model training
```bash
python src/run_model_training.py
```

6. Compute slice metrics
```bash
python src/compute_score.py
```

7. Run API tests
```bash
python python sanitycheck.pysrc/test_main.py

src/test_main.py
```

8. Serve the API on local
```bash
uvicorn app.main:app
```

## CI/CD
Every step and automated test in the [CI/CD pipeline](.github/workflows/fastapici.yml).