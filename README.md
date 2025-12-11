<img width="881" height="313" alt="image" src="https://github.com/user-attachments/assets/32084eba-97ba-4b58-9239-668c530c1af2" /># Optimize-Manufacturing-Operations-with-a-Predictive-Maintenance-Model
1. Data Exploration & Validation Strategy
•	Plot all sensor readings (temperature, vibration, pressure, current, etc.) over time and overlay known failure events.
o	This helps visually identify patterns, anomalies, and pre-failure trends.
•	Check for missing timestamps, duplicates, sudden jumps, or sensor drift.
•	Study the temporal behavior of machines (daily patterns, shift-wise variations, seasonal effects).
•	Use a time-aware validation strategy — traditional k-fold must not be used.
o	Apply TimeSeriesSplit, rolling-forward validation, or expanding-window validation to simulate real-world model deployment.
•	Ensure that validation data always occurs later in time than training data to prevent data leakage.
