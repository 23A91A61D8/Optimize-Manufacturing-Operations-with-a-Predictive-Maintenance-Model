import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

# 1. Create sample data
np.random.seed(0)
df = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='h'),  # lowercase 'h' for hourly frequency
    'sensor_reading': np.random.normal(loc=50, scale=5, size=1000),
    'failure_event': np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
})

# 2. Plot sensor readings with failure events highlighted
plt.figure(figsize=(15, 6))
sns.lineplot(data=df, x='timestamp', y='sensor_reading', label='Sensor Reading')

failures = df[df['failure_event'] == 1]
plt.scatter(failures['timestamp'], failures['sensor_reading'], color='red', label='Failure Event', s=50)

plt.title('Sensor Readings Over Time with Failure Events')
plt.xlabel('Timestamp')
plt.ylabel('Sensor Reading')
plt.legend()
plt.show()


# 3. Example TimeSeriesSplit usage for model validation

# For demonstration, let's pretend 'sensor_reading' is the feature and 'failure_event' is the label
X = df[['sensor_reading']]
y = df['failure_event']

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Fold {fold + 1}")
    print(f"Training set: {len(train_idx)} samples")
    print(f"Test set: {len(test_idx)} samples\n")
    
    # Here you would train your model, e.g.
    # model.fit(X_train, y_train)
    # preds = model.predict(X_test)
    # Evaluate model performance on test set

