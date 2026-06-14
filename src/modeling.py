"""
Model training, evaluation, and comparison utilities.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not installed. XGBoost models will be unavailable.")


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Evaluate regression model and return metrics dict.

    Returns
    -------
    dict
        Dictionary with MAE, MSE, RMSE, R2 scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Regression Results")
    print(f"{'='*50}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

    return metrics


def evaluate_classification(y_true, y_pred, model_name="Model"):
    """
    Evaluate classification model and print detailed results.

    Returns
    -------
    dict
        Dictionary with accuracy score.
    """
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Classification Results")
    print(f"{'='*50}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")

    return {"Model": model_name, "Accuracy": round(acc, 4)}


def train_rf_regressor(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgb_regressor(
    X_train, y_train,
    n_estimators=320, learning_rate=0.1, random_state=42,
):
    """Train an XGBoost Regressor."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is required. Install with: pip install xgboost")
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_rf_classifier(X_train, y_train, n_estimators=200, random_state=42):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgb_classifier(
    X_train, y_train,
    n_estimators=200, learning_rate=0.1, random_state=42,
):
    """Train an XGBoost Classifier."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is required. Install with: pip install xgboost")
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def tune_xgb_regressor(X_train, y_train, param_grid=None, cv=5):
    """
    Perform GridSearchCV for XGBoost Regressor hyperparameter tuning.

    Parameters
    ----------
    param_grid : dict, optional
        Parameter grid for search. Uses sensible defaults if not provided.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost is required.")

    default_grid = {
        "n_estimators": [200, 300, 400, 500],
        "learning_rate": [0.07, 0.1, 0.15],
        "max_depth": [3, 5, 6],
    }
    param_grid = param_grid or default_grid

    grid = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV R2:  {grid.best_score_:.4f}")

    return grid


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based model.

    Returns
    -------
    pd.DataFrame
        DataFrame with Feature and Importance columns, sorted descending.
    """
    imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    })
    return imp.sort_values("Importance", ascending=False).reset_index(drop=True)


def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Create a comparison table from a list of metric dictionaries.

    Parameters
    ----------
    metrics_list : list of dict
        Each dict should have at least 'Model' key and metric values.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    return pd.DataFrame(metrics_list).set_index("Model")
