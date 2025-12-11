# scripts/utils.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def load_csv(path):
    """
    Load data from CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(path)
    print(f"Loaded data from {path} with shape {df.shape}")
    return df

def save_model(model, path):
    """
    Save trained model to disk using joblib.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Load saved model from disk.
    """
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6,6)):
    """
    Plot confusion matrix using matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels if labels else ['Negative', 'Positive'])
    ax.set_yticklabels([''] + labels if labels else ['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center', color='red')
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    Print classification report.
    """
    report = classification_report(y_true, y_pred, zero_division=0)
    print("Classification Report:\n", report)

def set_plot_style():
    """
    Set a consistent style for matplotlib plots.
    """
    plt.style.use('seaborn-darkgrid')
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
