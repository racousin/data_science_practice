import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error


def compare_predictions(true_values_path, predictions_path):
    y_true = pd.read_csv(true_values_path)
    y_pred = pd.read_csv(predictions_path)
    error = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error: {error}")

    # Set a threshold for the error to trigger a test failure
    error_threshold = 12000

    # Raise an exception if the error exceeds the threshold
    if error > error_threshold:
        raise ValueError(
            f"Mean Absolute Error exceeds threshold: {error} > {error_threshold}"
        )


if __name__ == "__main__":
    true_values_path = sys.argv[1]
    predictions_path = sys.argv[2]
    compare_predictions(true_values_path, predictions_path)
