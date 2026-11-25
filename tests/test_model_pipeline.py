
import pandas as pd

from model_pipeline import train_model, evaluate_model


def test_train_model():

    X_train = pd.DataFrame({
        "Age": [25, 35, 45, 30, 50, 40],
        "Balance": [1000, 2000, 3000, 1500, 4000, 2500]
    })

    y_train = pd.Series([0, 1, 1, 0, 1, 0])

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")


def test_evaluate_model():

    X_test = pd.DataFrame({
        "Age": [25, 35, 45, 30],
        "Balance": [1000, 2000, 3000, 1500]
    })

    y_test = pd.Series([0, 1, 1, 0])

    model = train_model(X_test, y_test)

    accuracy, matrix = evaluate_model(
        model,
        X_test,
        y_test
    )

    assert 0 <= accuracy <= 1
    assert matrix.shape == (2, 2)

