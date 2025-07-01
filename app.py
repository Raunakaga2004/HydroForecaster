from flask import Flask, render_template, request
import pandas as pd
from utils.forecast import forecast_next_12_months
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dummy_groundwater_dataset.csv", parse_dates=["date"])
districts = sorted(df['district'].unique())

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_df = None
    selected_district = None

    if request.method == "POST":
        selected_district = request.form["district"]
        forecast_df = forecast_next_12_months(df, selected_district)

        # Save plot
        plt.figure(figsize=(10, 4))
        plt.plot(forecast_df["date"], forecast_df["predicted_groundwater_level"], marker="o")
        plt.title(f"LSTM Forecast - {selected_district}")
        plt.ylabel("Groundwater Level (m bgl)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/forecast_plot.png")
        plt.close()

    return render_template("index.html", districts=districts, forecast_df=forecast_df, selected=selected_district)

if __name__ == "__main__":
    app.run(debug=True)
