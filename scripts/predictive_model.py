# scripts/predictive_model.py

import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def train_xgb_with_cv(X, y, n_splits=5, random_state=42):
    """
    Train XGBoost classifier with TimeSeriesSplit cross-validation and SMOTE oversampling.
    Prints classification reports and plots Precision-Recall and ROC curves for each fold.
    
    Parameters:
        X (pd.DataFrame or np.array): Feature data.
        y (pd.Series or np.array): Target labels.
        n_splits (int): Number of CV splits.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        model (xgb.XGBClassifier): The model trained on the last fold.
    """
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Apply SMOTE on training data only
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Initialize XGBoost classifier
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=random_state
        )
        
        model.fit(X_train_res, y_train_res)
        
        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)[:, 1]
        
        print(f"Fold {fold + 1} Classification Report:")
        print(classification_report(y_test, preds, zero_division=0))
        
        avg_precision = average_precision_score(y_test, pred_probs)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        print(f"Fold {fold + 1} Average Precision (AP): {avg_precision:.4f}")
        print(f"Fold {fold + 1} F1-score: {f1:.4f}")
        
        # Plot Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, pred_probs)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f'Fold {fold + 1} Precision-Recall')
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
    
    # After all folds, plot feature importance for last fold model
    xgb.plot_importance(model, max_num_features=10)
    plt.title('Feature Importance')
    plt.show()
    
    return model

def predict(model, X_new):
    """
    Predict using the trained model.
    
    Parameters:
        model (xgb.XGBClassifier): Trained model.
        X_new (pd.DataFrame or np.array): New feature data.
    
    Returns:
        preds (np.array): Predicted class labels.
        pred_probs (np.array): Predicted probabilities for positive class.
    """
    preds = model.predict(X_new)
    pred_probs = model.predict_proba(X_new)[:, 1]
    return preds, pred_probs
