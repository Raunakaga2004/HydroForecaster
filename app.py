from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from utils.preprocessing import GroundwaterPreprocessor

app = Flask(__name__)

# Load LSTM model and preprocessing pipeline
model = load_model('models/lstm_model.h5', compile=False)
preprocessor = joblib.load('preprocess.pkl')

# API details
API_URL = "https://ckandev.indiadataportal.com/api/action/datastore_search_sql"
RESOURCE_ID = "580a8f6e-3d86-4ca7-ac7d-cd5df12b443c"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/locations')
def locations():
    sql_districts = f'''SELECT DISTINCT "district_name" FROM "{RESOURCE_ID}" ORDER BY "district_name"'''
    sql_stations = f'''SELECT DISTINCT "station_name", "district_name" FROM "{RESOURCE_ID}" ORDER BY "station_name"'''

    headers = {'Content-Type': 'application/json'}
    response_districts = requests.post(API_URL, headers=headers, json={"sql": sql_districts})
    response_stations = requests.post(API_URL, headers=headers, json={"sql": sql_stations})

    if response_districts.status_code != 200 or response_stations.status_code != 200:
        return jsonify({"districts": [], "stations": []})

    districts = [rec['district_name'] for rec in response_districts.json().get('result', {}).get('records', [])]
    stations_data = response_stations.json().get('result', {}).get('records', [])

    district_station_map = {}
    for rec in stations_data:
        district = rec['district_name']
        station = rec['station_name']
        district_station_map.setdefault(district, []).append(station)

    district_query = request.args.get('district')
    if district_query:
        stations = district_station_map.get(district_query, [])
        return jsonify({"stations": stations})

    return jsonify({"districts": districts})

@app.route('/predict')
def predict_latest():
    district = request.args.get('district')
    station = request.args.get('station')

    sql_query = f'''SELECT * FROM "{RESOURCE_ID}" WHERE "currentlevel" IS NOT NULL'''
    if district:
        sql_query += f''' AND "district_name" = '{district}' '''
    if station:
        sql_query += f''' AND "station_name" = '{station}' '''
    sql_query += ''' ORDER BY "date" DESC LIMIT 50'''

    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, headers=headers, json={"sql": sql_query})
    if response.status_code != 200:
        return render_template('result.html', prediction="API Error", location="Unknown")

    records = response.json().get('result', {}).get('records', [])
    if not records or len(records) < 10:
        return render_template('result.html', prediction="Not enough data", location="Unknown")

    df_latest = pd.DataFrame(records)

    # Convert to datetime
    df_latest['date'] = pd.to_datetime(df_latest['date'], errors='coerce')
    df_latest.dropna(subset=['date'], inplace=True)

    df_processed = preprocessor.transform(df_latest)
    df_processed = df_processed[['currentlevel', 'level_diff', 'season']]

    required_steps = 10
    actual_steps = df_processed.shape[0]

    if actual_steps < required_steps:
        pad_rows = required_steps - actual_steps
        padding = np.zeros((pad_rows, df_processed.shape[1]))
        padded_input = np.vstack([padding, df_processed.values])
    else:
        padded_input = df_processed.tail(required_steps).values

    X_input = padded_input.reshape(1, required_steps, df_processed.shape[1])

    # Multi-step forecasting (next 7 days)
    future_steps = 7
    future_preds = []
    input_seq = X_input.copy()

    for _ in range(future_steps):
        next_pred = model.predict(input_seq, verbose=0)[0][0]
        future_preds.append(round(next_pred, 2))

        last_known = input_seq[0, -1, :].copy()
        new_row = last_known.copy()
        new_row[0] = next_pred
        new_row[1] = next_pred - last_known[0]
        input_seq = np.append(input_seq[:, 1:, :], [[new_row]], axis=1)

    location = df_latest['location'].iloc[0] if 'location' in df_latest.columns else "Unknown"

    # Plotting
    plt.figure(figsize=(10, 4))
    df_latest_sorted = df_latest.sort_values('date')
    plt.plot(df_latest_sorted['date'], df_latest_sorted['currentlevel'], marker='o', label='Past')

    # Generate future dates
    start_date = pd.to_datetime(df_latest_sorted['date'].max()) + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=future_steps)

    plt.plot(future_dates, future_preds, marker='x', linestyle='--', color='orange', label='Predicted')
    plt.title('Groundwater Levels Forecast')
    plt.xlabel('Date')
    plt.ylabel('Current Level')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template(
        'result.html',
        prediction=future_preds,
        location=location,
        table=df_latest_sorted[['date', 'district_name', 'station_name', 'currentlevel']].to_html(classes='table table-bordered', index=False),
        plot_url=plot_url
    )

if __name__ == '__main__':
    app.run(debug=True)
