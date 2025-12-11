df['rolling_std_3'] = df['sensor_reading'].rolling(window=3).std().bfill()
df['ema_5'] = df['sensor_reading'].ewm(span=5, adjust=False).mean()
df['gradient'] = np.gradient(df['sensor_reading'])
df['time_since_maintenance'] = np.random.randint(1, 100, size=len(df))
df['operational_age'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600

print(df.head(10))  # <-- Add this line to see output
