# models/predictive_model.py

import xgboost as xgb
import shap
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """
    Train and cache an XGBoost classifier model.
    """
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def get_shap_values(model: xgb.XGBClassifier, data: pd.DataFrame):
    """
    Calculate SHAP values for given data using the model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    return shap_values
