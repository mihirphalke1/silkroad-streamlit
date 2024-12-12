from prophet import Prophet
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib  # For saving and loading models

# Load the data
df = pd.read_csv('data/feature_engineered_data.csv')
df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])

# Aggregate the data by month and medicine name
df['Month'] = df['Expiry Date'].dt.to_period('M')
monthly_demand = df.groupby(['Month', 'Name'])['Quantity of Order'].sum().unstack().fillna(0)

# Create output directories for saving plots and models
output_dir = "plots/future_demand"
models_dir = "models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Current date and end date for the next 12 months
current_date = datetime.now()
end_date = current_date + timedelta(days=365)

for medicine in monthly_demand.columns:
    print(f"Processing {medicine}...")

    # Prepare data for Prophet
    data = monthly_demand[[medicine]].reset_index()
    data.columns = ['ds', 'y']  # Prophet requires specific column names
    data['ds'] = data['ds'].dt.to_timestamp()  # Ensure datetime format

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(data)

    # Save the trained model to a `.pkl` file
    model_path = os.path.join(models_dir, f"{medicine}_model.pkl")
    joblib.dump(model, model_path)

    # Make future dataframe for predictions
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # Filter forecast to include only the next 12 months
    forecast_filtered = forecast[forecast['ds'] <= end_date]

    # Plot forecast
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(data['ds'], data['y'], label='Historical Demand')
    ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label='Forecast', linestyle='--')
    ax.fill_between(
        forecast_filtered['ds'],
        forecast_filtered['yhat_lower'],
        forecast_filtered['yhat_upper'],
        color='gray',
        alpha=0.2,
        label='Confidence Interval',
    )

    # Customize plot
    ax.set_title(f'Demand Forecast for {medicine} (From {current_date.strftime("%Y-%m")} for 12 Months)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity of Order')
    ax.legend()
    ax.grid()

    # Set X-axis limits to match the 12-month range
    ax.set_xlim([current_date, end_date])

    # Save the plot
    plt.savefig(f"{output_dir}/{medicine}_future_demand.png")
    plt.close()

print(f"Forecast plots saved in '{output_dir}' and models saved in '{models_dir}' directories.")