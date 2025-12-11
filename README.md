# Optimize-Manufacturing-Operations-with-a-Predictive-Maintenance-Model
## 1. Data Exploration & Validation Strategy

- **Plot all sensor readings** (temperature, vibration, pressure, current, etc.) over time and overlay known failure events.  
  - This helps visually identify patterns, anomalies, and pre-failure trends.

- **Check for missing timestamps, duplicates, sudden jumps, or sensor drift.**

- **Study temporal machine behavior**, including:
  - Daily patterns  
  - Shift-wise variations  
  - Seasonal effects  

- **Use a time-aware validation strategy** — traditional k-fold must not be used.  
  - Apply **TimeSeriesSplit**, **rolling-forward validation**, or **expanding-window validation** to simulate real-world model deployment.

- **Ensure validation data always occurs later in time than training data** to prevent data leakage.

<img width="881" height="313" alt="image" src="https://github.com/user-attachments/assets/97701f2d-27a0-4fd0-8447-4138959b5576" />
<img width="935" height="492" alt="image" src="https://github.com/user-attachments/assets/6bb2f2c0-94d8-40cf-8146-2f081238b15f" />

## 2. Feature Engineering

- Create features that capture trends, volatility, and time-based patterns in signals.

- **Strong feature categories include:**

  - **Rolling statistics**
    - Rolling mean, rolling standard deviation  
    - Rolling min/max  
    - Rolling median  
    - Common window sizes: **1 hr, 3 hr, 6 hr, 12 hr**

  - **Exponential Moving Averages (EMA)**
    - Helps smooth noisy sensor readings.

  - **Gradient / Rate-of-change**
    - First derivative  
    - Second derivative  
    - Useful for detecting rising vibration/temperature before failure.

  - **Operational features**
    - Time since last maintenance  
    - Machine age  
    - Shift number (morning / evening / night)

- Ensure **consistent lookback windows**, so the model always receives a fixed-size feature set for each timestamp.

<img width="936" height="594" alt="image" src="https://github.com/user-attachments/assets/8b0cebb3-0421-4341-8dc2-09c76607bdf1" />

## 3. Modeling for Imbalanced Data

- Failure data is naturally imbalanced; only a small percentage of timestamps correspond to failures.

- **Apply class imbalance handling:**
  - Use `scale_pos_weight` in XGBoost/LightGBM  
  - Or `class_weight='balanced'` in RandomForest or Logistic Regression

- **Optional:** Test oversampling methods like SMOTE  
  - Apply **only inside training folds**  
  - Never oversample across time boundaries

- **Primary evaluation metrics should be:**
  - Precision–Recall Curve  
  - F1-score  
  - PR-AUC (Precision–Recall AUC)

- Avoid using **accuracy**, as it gives misleading results for imbalanced problems.

- Tune decision thresholds based on operational needs:  
  - High **recall** → safety-critical systems  
  - High **precision** → reduce false alarms

<img width="1067" height="145" alt="Screenshot 2025-12-11 194656" src="https://github.com/user-attachments/assets/4e3b7c38-a34f-4d48-ba22-d88dff8328a0" />
<img width="508" height="523" alt="image" src="https://github.com/user-attachments/assets/019d9cd5-8c1c-4c28-b100-ac6dab1e9b68" />
<img width="411" height="523" alt="image" src="https://github.com/user-attachments/assets/2625804e-827c-408a-a34e-43a69a475123" />
<img width="465" height="455" alt="image" src="https://github.com/user-attachments/assets/64c94ed7-783a-4b0f-9c4a-6da3e18a0273" />
<img width="460" height="450" alt="image" src="https://github.com/user-attachments/assets/9e69b5c5-01a4-4861-8ef2-e96d12db9a32" />
<img width="524" height="552" alt="image" src="https://github.com/user-attachments/assets/fdf29274-acf4-4ed9-90ab-578f408da05d" />
<img width="404" height="518" alt="image" src="https://github.com/user-attachments/assets/5651de4e-4980-4e39-a2df-b9a26c748f27" />
<img width="439" height="450" alt="image" src="https://github.com/user-attachments/assets/4767cb3c-8932-4855-904e-db58867edeef" />
<img width="493" height="428" alt="image" src="https://github.com/user-attachments/assets/b5fcec6f-e9f0-4b57-a8f7-3982127d680d" />
<img width="460" height="481" alt="image" src="https://github.com/user-attachments/assets/7bb37983-7db0-4d43-8f3c-5ef9f4a65c6c" />
<img width="458" height="478" alt="image" src="https://github.com/user-attachments/assets/1e558993-0be2-4112-b1b2-5ad14d7e3e61" />
<img width="689" height="362" alt="image" src="https://github.com/user-attachments/assets/cf977c11-7902-47f7-a653-9497dad92b20" />
<img width="916" height="957" alt="image" src="https://github.com/user-attachments/assets/d29ab5c6-adf4-4ec9-b584-2fbc0599e9c7" />

## 4. Dashboard Development

- The dashboard should be designed with **real end-users in mind**, especially maintenance engineers who monitor equipment conditions.

- **Main dashboard should include:**
  - Total machine count  
  - Failure count  
  - Failure rate (%)
  - List of high-risk machines  
  - Machine selector (dropdown menu)

- **For each selected machine, display:**
  - Historical sensor readings with failure event markers  
  - Predicted failure risk score (numeric value or gauge indicator)  
  - SHAP-based explanation of the top contributing factors  

- **Include interactive features such as:**
  - Zoomable sensor charts  
  - Hover tooltips to inspect values  
  - Dynamic risk trend plots  

- Ensure that the UI is **clear, responsive**, and helps users quickly identify **machines that are at risk**, enabling faster decision-making.

<img width="798" height="885" alt="image" src="https://github.com/user-attachments/assets/fc14377b-0a10-435d-92ab-38c3078cb4c0" />
<img width="833" height="1193" alt="image" src="https://github.com/user-attachments/assets/f8daec94-f573-4e77-b962-9304b630474b" />
<img width="781" height="801" alt="image" src="https://github.com/user-attachments/assets/9469468c-230e-4ed2-b315-07a3e1a8e5d0" />

## 5. Project Structure & Reproducibility

- **Organize the project clearly to ensure repeatability:**
  - Separate directories for raw data, processed data, notebooks, scripts, and dashboards.

- **Maintain a `requirements.txt` file to fix library versions.**

- **Use consistent random seeds for reproducibility in all notebooks and model scripts.**

- **Maintain a clear folder for experiment logs, evaluation metrics, and model artifacts.**

- **Export the final trained model and ensure the dashboard loads it correctly each time.**

<img width="486" height="1168" alt="image" src="https://github.com/user-attachments/assets/c76ddd56-8765-44c9-a5ea-c206cb9ca51b" />
<img width="360" height="1167" alt="image" src="https://github.com/user-attachments/assets/fb3e5fd9-5ff4-41d1-95f5-973b737cc754" />

## Random Seed Initialization

- To ensure that all experiments in the project produce consistent and repeatable results, random seeds are initialized at the beginning of the workflow.

- Machine learning pipelines rely on randomness in several stages:
  - Train/validation splitting
  - Model weight initialization
  - Feature shuffling
  - Sampling
  - Any NumPy or Python-based random operations

- If the random seed is not fixed, you will get different results every time, which makes debugging and comparing models impossible.
## Code Used for Random Seed Initialization
<img width="940" height="258" alt="image" src="https://github.com/user-attachments/assets/5e1486d0-b604-496d-8d84-043c8303222f" />





