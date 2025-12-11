import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def train_evaluate(X, y, n_splits=5, random_state=42):
    """
    Train and evaluate an XGBoost classifier using TimeSeriesSplit cross-validation.
    
    Args:
        X (pd.DataFrame or np.array): Features.
        y (pd.Series or np.array): Target variable (binary).
        n_splits (int): Number of TimeSeriesSplit folds.
        random_state (int): Seed for reproducibility.
    """
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"Scale positive weight: {scale_pos_weight:.2f}\n")
    
    fold_models = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"=== Fold {fold + 1} ===")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Handle class imbalance using SMOTE on training set only
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=random_state
        )
        model.fit(X_train_res, y_train_res)

        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)[:, 1]

        print("Classification Report:")
        print(classification_report(y_test, preds, zero_division=0))

        avg_precision = average_precision_score(y_test, pred_probs)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"Average Precision (AP): {avg_precision:.4f}")
        print(f"F1-score: {f1:.4f}\n")

        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, pred_probs)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f'Fold {fold + 1}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Fold {fold + 1}')
        plt.legend()
        plt.show()

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'Fold {fold + 1} ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold + 1}')
        plt.legend()
        plt.show()

        fold_models.append(model)
    
    # Show feature importance from the last fold model
    print("Feature importance from last fold:")
    xgb.plot_importance(fold_models[-1], max_num_features=10)
    plt.show()

    return fold_models

if __name__ == "__main__":
    # Example: load preprocessed data
    # Replace this with your own data loading logic

    print("Loading preprocessed data...")

    X = pd.read_csv("processed_features.csv")  # Your processed features CSV
    y = pd.read_csv("processed_target.csv")['target']  # Your target CSV

    print(f"Loaded features shape: {X.shape}, target shape: {y.shape}\n")

    # Run training and evaluation
    train_evaluate(X, y)
