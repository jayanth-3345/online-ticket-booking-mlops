
cat > README.md <<'EOF'
# Online Ticket Booking System - MLOps

## 1. Project Overview

This project extends the Software Engineering Online Ticket Booking System with a machine learning component and an automated MLOps pipeline.

The booking system requirements include:
- User registration and login
- Movie search
- Seat selection
- Online payment
- Booking confirmation
- Admin movie management

The machine learning component demonstrates movie-rating prediction using K-Nearest Neighbors (KNN).

## 2. Machine Learning Component

### Algorithm
K-Nearest Neighbors (KNN) Regression.

### Features
- User ID
- Movie ID

### Target
- Movie rating

### Dataset
The dataset is `data/movies.csv`. DVC tracks the dataset version, while the actual dataset is stored in a Google Drive remote.

### Model Configuration
- Algorithm: KNN Regression
- Number of neighbors: 3
- Train/test split: 80% / 20%
- Random state: 42
- Mean Squared Error (MSE): 0.3111 (recorded local run)

## 3. Technologies Used

- Python
- scikit-learn
- pandas
- Git and GitHub
- DVC
- Google Drive
- MLflow
- GitHub Actions
- pytest
- joblib

## 4. MLOps Pipeline

The GitHub Actions workflow runs on pushes to `main` and pull requests targeting `main`.

Pipeline steps:
1. Check out the repository.
2. Set up Python.
3. Install project dependencies and DVC Google Drive support.
4. Configure DVC service-account authentication.
5. Pull the versioned dataset using `dvc pull`.
6. Train the KNN model.
7. Log model parameters, MSE, and the model to MLflow.
8. Run automated tests using pytest.
9. Upload the trained model as a GitHub Actions artifact.

## 5. Data Version Control (DVC)

DVC tracks the dataset without storing the dataset contents directly in Git.

Tracked metadata:
`data/movies.csv.dvc`

To retrieve the dataset after configuring access to the remote:

```bash
dvc pull
```

The remote storage is Google Drive. GitHub Actions authenticates through the repository secret `GDRIVE_CREDENTIALS_DATA`. The service-account JSON key must remain private and must not be committed to the repository.

## 6. MLflow Experiment Tracking

The training script uses the MLflow experiment:

`movie-recommendation`

It logs:
- Algorithm name
- Number of neighbors
- Mean Squared Error
- Trained model

Run training locally with:

```bash
python src/train.py
```

## 7. Continuous Integration

GitHub Actions automatically runs the workflow when code is pushed to `main` or a pull request targets `main`.

The workflow checks out the code, installs dependencies, retrieves the dataset, trains the model, and runs tests.

## 8. Continuous Build

The model training command is:

```bash
python src/train.py
```

The trained model is saved locally to:

`models/model.joblib`

The workflow uploads this file as the artifact:

`movie-recommendation-model`

The artifact can be downloaded from the successful GitHub Actions run.

## 9. Continuous Testing

Tests are implemented with pytest.

Run them locally using:

```bash
pytest -v
```

The current test suite contains four tests covering:
- Dataset is not empty
- Required dataset columns exist
- Model training works
- Model performance meets the configured MSE threshold

Latest verified local result: **4 tests passed**.

## 10. Project Structure

```text
online-ticket-booking-mlops/
├── data/
│   └── movies.csv.dvc
├── models/
│   └── model.joblib
├── src/
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── .github/
│   └── workflows/
│       └── mlops.yml
├── params.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

## 11. Run Locally

Activate the virtual environment:

```bash
source venv/bin/activate
```

Retrieve the dataset if needed and remote access is configured:

```bash
dvc pull
```

Train the model:

```bash
python src/train.py
```

Run the tests:

```bash
pytest -v
```

## 12. Result

The GitHub Actions pipeline has successfully:
- Retrieved the DVC-managed dataset from Google Drive
- Trained the KNN model
- Run the automated test suite
- Uploaded the trained model as a downloadable artifact

The project demonstrates dataset versioning with DVC, experiment tracking with MLflow, and automated integration, model building, and testing through GitHub Actions.
