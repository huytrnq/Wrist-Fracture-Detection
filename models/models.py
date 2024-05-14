from joblib import load
from joblib import dump

def export_model(model, filename):
    """
    Exports a trained scikit-learn model to a file using joblib.

    Args:
    - model (sklearn.base.BaseEstimator): The trained model to export.
    - filename (str): The path to the file where the model will be saved.
    """
    dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Loads a scikit-learn model from a file using joblib.

    Args:
    - filename (str): The path to the file from which the model will be loaded.

    Returns:
    - model (sklearn.base.BaseEstimator): The loaded scikit-learn model.
    """
    model = load(filename)
    print(f"Model loaded from {filename}")
    return model
