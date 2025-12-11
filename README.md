# Optimize-Manufacturing-Operations-with-a-Predictive-Maintenance-Model
1. Data Exploration & Validation Strategy
•	Plot all sensor readings (temperature, vibration, pressure, current, etc.) over time and overlay known failure events.
o	This helps visually identify patterns, anomalies, and pre-failure trends.
•	Check for missing timestamps, duplicates, sudden jumps, or sensor drift.
•	Study the temporal behavior of machines (daily patterns, shift-wise variations, seasonal effects).
•	Use a time-aware validation strategy — traditional k-fold must not be used.
o	Apply TimeSeriesSplit, rolling-forward validation, or expanding-window validation to simulate real-world model deployment.
•	Ensure that validation data always occurs later in time than training data to prevent data leakage.
<img width="881" height="313" alt="image" src="https://github.com/user-attachments/assets/97701f2d-27a0-4fd0-8447-4138959b5576" />
<img width="935" height="492" alt="image" src="https://github.com/user-attachments/assets/6bb2f2c0-94d8-40cf-8146-2f081238b15f" />

2. Feature Engineering
•	Create features that capture trends, volatility, and time-based patterns in signals.
•	Strong features commonly include:
o	Rolling statistics:
	Rolling mean, rolling std deviation
	Rolling min/max
	Rolling median
	Useful windows: 1 hr, 3 hr, 6 hr, 12 hr
o	Exponential moving averages (EMA) to smooth noisy sensor readings.
o	Gradient / Rate-of-change:
	First derivative, second derivative
	Helps detect increasing vibration/temperature before failure
o	Operational features:
	Time since last maintenance
	Machine age
	Shift number (morning/evening/night)
•	Capture consistent lookback windows so the model sees fixed-size feature sets for each timestamp.
<img width="936" height="594" alt="image" src="https://github.com/user-attachments/assets/8b0cebb3-0421-4341-8dc2-09c76607bdf1" />

3. Modeling for Imbalanced Data
•	Failure data is naturally imbalanced; only a small percentage of timestamps correspond to failures.
•	Apply class imbalance handling:
o	Use scale_pos_weight in XGBoost/LightGBM
o	Or class_weight='balanced' in RandomForest or Logistic Regression
•	Optionally test oversampling methods like SMOTE — only inside training folds, never across time boundaries.
•	Primary evaluation metrics should be:
o	Precision-Recall Curve
o	F1-score
o	PR-AUC (Precision–Recall AUC)
•	Avoid accuracy as it gives misleading results for imbalanced problems.
•	Tune decision thresholds for your operational needs (high recall for safety, high precision for reducing false alarms).
<img width="508" height="523" alt="image" src="https://github.com/user-attachments/assets/019d9cd5-8c1c-4c28-b100-ac6dab1e9b68" />     <img width="411" height="523" alt="image" src="https://github.com/user-attachments/assets/2625804e-827c-408a-a34e-43a69a475123" />




