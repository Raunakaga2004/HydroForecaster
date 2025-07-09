from flask import Flask, render_template
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
model = load_model('models/lstm_model.h5', compile = False)
preprocessor = joblib.load('preprocess.pkl')

# API details
API_URL = "https://ckandev.indiadataportal.com/api/action/datastore_search_sql"
RESOURCE_ID = "580a8f6e-3d86-4ca7-ac7d-cd5df12b443c"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_latest():
    # SQL query to get the latest 10 records
    sql_query = f'''
        SELECT * FROM "{RESOURCE_ID}"
        WHERE "currentlevel" IS NOT NULL
        ORDER BY "date" DESC
        LIMIT 50
    '''

    headers = {'Content-Type': 'application/json'}

    response = requests.post(API_URL, headers=headers, json={"sql": sql_query})
    if response.status_code != 200:
        return render_template('result.html', prediction="API Error", location="Unknown")

    records = response.json().get('result', {}).get('records', [])
    if not records or len(records) < 10:
        return render_template('result.html', prediction="Not enough data", location="Unknown")

    # Convert to DataFrame
    df_latest = pd.DataFrame(records)
    print(df_latest.district_name)

    # After transforming the latest data
    df_processed = preprocessor.transform(df_latest)
    df_processed = df_processed[['currentlevel', 'level_diff', 'season']]  # or the exact 3 used

    # print(df_processed)

    print("✅ Final shape after preprocessing:", df_processed.shape)

    # Required time steps
    required_steps = 10
    actual_steps = df_processed.shape[0]

    # If not enough rows, pad with zeros
    if actual_steps < required_steps:
        print(f"⚠️ Only {actual_steps} time steps found. Padding with {required_steps - actual_steps} zeros.")
        
        pad_rows = required_steps - actual_steps
        padding = np.zeros((pad_rows, df_processed.shape[1]))  # same number of features
        padded_input = np.vstack([padding, df_processed.values])  # stack zeros at the beginning
    else:
        padded_input = df_processed.tail(required_steps).values

    # Reshape to (1, 10, features) for LSTM
    X_input = padded_input.reshape(1, required_steps, df_processed.shape[1])

    # Predict
    prediction = model.predict(X_input)[0][0]

    # Location safe access
    location = df_latest['location'].iloc[0] if 'location' in df_latest.columns else "Unknown"
    
    
    # Plot recent current levels
    plt.figure(figsize=(8, 4))
    df_latest_sorted = df_latest.sort_values('date')
    plt.plot(df_latest_sorted['date'], df_latest_sorted['currentlevel'], marker='o', color='teal')
    plt.title('Recent Groundwater Levels')
    plt.xlabel('Date')
    plt.ylabel('Current Level')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Close the plot to avoid memory leak
    plt.close()

    # Pass prediction, location, plot, and table to result.html
    return render_template(
        'result.html',
        prediction=round(prediction, 2),
        location=location,
        table=df_latest_sorted[['date', 'district_name', 'station_name', 'currentlevel']].to_html(classes='table table-bordered', index=False),
        plot_url=plot_url
    )


    # return render_template('result.html', prediction=round(prediction, 2), location=location)

if __name__ == '__main__':
    app.run(debug=True)
