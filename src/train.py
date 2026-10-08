import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load dataset
data = pd.read_csv("data/movies.csv")

# Convert user/movie IDs into features
X = data[["user_id", "movie_id"]]
y = data["rating"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model parameters
n_neighbors = 3

# Start MLflow experiment
mlflow.set_experiment("movie-recommendation")

with mlflow.start_run():

    # Create model
    model = KNeighborsRegressor(n_neighbors=n_neighbors)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate error
    mse = mean_squared_error(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("algorithm", "KNN")
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_metric("mse", mse)

    # Save model for Continuous Build
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")

    # Save model to MLflow
    mlflow.sklearn.log_model(
        model,
        "model",
        skops_trusted_types=[
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
            "sklearn.neighbors._kd_tree.KDTree",
        ],
    )

print("Movie Recommendation Model")
print("--------------------------")
print("Algorithm: KNN")
print(f"Neighbors: {n_neighbors}")
print(f"MSE: {mse:.4f}")
print("Model saved to models/model.joblib")
print("Model logged to MLflow successfully.")
