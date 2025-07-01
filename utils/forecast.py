import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta

def forecast_next_12_months(df, district_name):
    # Load model
    model = load_model("models/lstm_model.h5", compile=False)

    # Filter district data
    district_df = df[df['district'] == district_name].sort_values("date")

    # Select features
    features = ["rainfall_mm", "temperature_celsius", "groundwater_extraction_volume"]
    target = "groundwater_level_mbgl"
    data = district_df[features + [target]]

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Recursive prediction
    last_sequence = scaled_data[-12:, :-1]
    current_sequence = last_sequence.copy()
    predictions = []

    for _ in range(12):
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        next_pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(next_pred_scaled)
        next_features = current_sequence[-1].copy()
        current_sequence = np.vstack([current_sequence, next_features])[1:]

    # Inverse transform
    X_last = np.repeat(current_sequence[-1].reshape(1, -1), 12, axis=0)
    y_pred_scaled = np.array(predictions).reshape(-1, 1)
    full_scaled = np.hstack((X_last, y_pred_scaled))
    predicted_levels = scaler.inverse_transform(full_scaled)[:, -1]

    # Dates
    last_date = district_df['date'].max()
    future_dates = [last_date + relativedelta(months=i+1) for i in range(12)]

    return pd.DataFrame({
        'date': future_dates,
        'predicted_groundwater_level': predicted_levels
    })
