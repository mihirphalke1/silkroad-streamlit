import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle  # For saving and loading .pkl files

# Load the dataset
data = pd.read_csv('data/feature_engineered_data.csv')

# Ensure 'Order Date' is in datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Set 'Order Date' as the index
data.set_index('Order Date', inplace=True)

# Resample the data to a monthly frequency, summing quantities
monthly_data = data.resample('M').sum()

# Output directories for saving models and plots
models_dir = 'models/spike'
plots_dir = 'plots/spike'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Function to forecast demand for a specific medicine and save results as .pkl
def forecast_demand(medicine_name, medicine_data, forecast_periods=12):
    # Ensure the medicine name is safe for filenames
    safe_medicine_name = "".join(c if c.isalnum() else "_" for c in medicine_name)
    
    # Check if there are enough observations
    if len(medicine_data) < 12:
        print(f"Not enough data to forecast for Medicine Name: {medicine_name}. Available records: {len(medicine_data)}.")
        return
    
    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(medicine_data, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    
    # Forecast the future demand
    forecast = model_fit.forecast(steps=forecast_periods)
    
    # Combine historical data and forecast for plotting
    forecast_index = pd.date_range(start=medicine_data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='M')
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    # Save the fitted model to a .pkl file
    model_path = os.path.join(models_dir, f"{safe_medicine_name}_model.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(model_fit, model_file)
    
    # Save historical data, forecast, and spikes in a .pkl file
    spikes = medicine_data[medicine_data > medicine_data.mean() + 2 * medicine_data.std()]
    data_pkl = {
        "historical": medicine_data,
        "forecast": forecast_series,
        "spikes": spikes,
    }
    data_pkl_path = os.path.join(models_dir, f"{safe_medicine_name}_data.pkl")
    with open(data_pkl_path, "wb") as data_file:
        pickle.dump(data_pkl, data_file)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(medicine_data.index, medicine_data, label='Demand', color='blue')
    plt.scatter(spikes.index, spikes, color='orange', label='Spikes in Demand', marker='o')
    plt.plot(forecast_series.index, forecast_series, label='Forecast', color='green', linestyle='--')
    
    # Add title and labels
    plt.title(f'Demand Spike Analysis for Medicine: {medicine_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity of Order')
    plt.legend()
    plt.grid()
    
    # Save the plot as a .png file
    plot_path = os.path.join(plots_dir, f"{safe_medicine_name}_spike_analysis.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Forecast and plot for Medicine Name {medicine_name} saved successfully.")
    
    # Return the paths of saved files for further reference
    return {
        "medicine_name": medicine_name,
        "model_file": model_path,
        "data_file": data_pkl_path,
        "plot_file": plot_path
    }

# Generate forecasts and save results for all medicines
results = []
unique_medicines = data['Name'].unique()
for medicine_name in unique_medicines:
    print(f"Generating forecast plot for Medicine Name: {medicine_name}")
    medicine_data = data[data['Name'] == medicine_name]['Quantity of Order']
    result_data = forecast_demand(medicine_name=medicine_name, medicine_data=medicine_data)
    if result_data:
        results.append(result_data)

# Save all results (paths) in a .pkl file
overall_results_path = os.path.join(models_dir, "overall_results.pkl")
with open(overall_results_path, "wb") as results_file:
    pickle.dump(results, results_file)

print("Forecasts, plots, models, and overall results have been saved.")