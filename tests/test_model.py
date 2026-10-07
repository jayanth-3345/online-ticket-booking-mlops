import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def load_data():
    return pd.read_csv("data/movies.csv")


def test_dataset_not_empty():
    data = load_data()
    assert not data.empty


def test_required_columns():
    data = load_data()

    required_columns = {
        "user_id",
        "movie_id",
        "movie",
        "genre",
        "rating",
    }

    assert required_columns.issubset(data.columns)


def test_model_training():
    data = load_data()

    X = data[["user_id", "movie_id"]]
    y = data["rating"]

    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)

    assert model is not None


def test_model_performance():
    data = load_data()

    X = data[["user_id", "movie_id"]]
    y = data["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    # Continuous testing threshold
    assert mse < 1.0
