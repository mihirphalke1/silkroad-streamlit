import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle  # For saving and loading .pkl files
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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

output_dir = "outputs_expiry"
os.makedirs(output_dir, exist_ok=True)

plt.style.use('ggplot')  # Use a clean Matplotlib style

try:
    # Load the dataset
    file_path = "data/feature_engineered_data_with_ids.csv"
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # Data Preprocessing
    date_cols = ['Order Date', 'Expiry Date', 'Sale Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calculate time-based features
    if 'Expiry Date' in df.columns and 'Order Date' in df.columns:
        df['Days_to_Expiry'] = (df['Expiry Date'] - df['Order Date']).dt.days
    else:
        raise ValueError("Missing columns required to calculate 'Days_to_Expiry'.")

    if 'Sale Date' in df.columns and 'Order Date' in df.columns:
        df['Time_Since_Sale'] = (df['Sale Date'] - df['Order Date']).dt.days
    else:
        df['Time_Since_Sale'] = np.nan

    if 'Sales Quantity' in df.columns and 'Stock at Distributor' in df.columns:
        df['Stock_Turnover_Rate'] = df['Sales Quantity'] / (df['Stock at Distributor'] + 1e-9)
    else:
        df['Stock_Turnover_Rate'] = np.nan

    # Location feature engineering
    location_features = [
        'Current Stage Location', 'Retailer Location', 'Distributor Location', 
        'Manufacturer Location', 'Raw Material Supplier Location'
    ]
    available_location_features = [feat for feat in location_features if feat in df.columns]

    if available_location_features:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_locations = ohe.fit_transform(df[available_location_features].fillna('missing'))
        encoded_df = pd.DataFrame(encoded_locations, 
                                  columns=ohe.get_feature_names_out(available_location_features))
        df = pd.concat([df, encoded_df], axis=1)
    else:
        encoded_df = pd.DataFrame()

    # Define and prepare features
    numerical_features = [
        'Time_Since_Order', 'Days in Current Stage', 'Quantity of Order', 
        'Stock at Distributor', 'Lead Time', 'Expiry Risk Score', 'Anomaly Score',
        'Turn Over Ratio', 'Supply Chain Velocity', 'Stage Transition Time', 
        'Time to Critical', 'Stock_Turnover_Rate', 'Time_Since_Sale'
    ]
    available_numerical_features = [feat for feat in numerical_features if feat in df.columns]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[available_numerical_features] = imputer.fit_transform(df[available_numerical_features])
    df.dropna(subset=['Days_to_Expiry'], inplace=True)

    # Prepare features and target
    X = df[available_numerical_features + list(encoded_df.columns)]
    y = df['Days_to_Expiry']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model with optimized parameters
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    # Make predictions and evaluate
    predictions = model.predict(X_scaled)  # Using training data for evaluation
    metrics = {
        'MSE': mean_squared_error(y, predictions),
        'RMSE': np.sqrt(mean_squared_error(y, predictions)),
        'MAE': mean_absolute_error(y, predictions),
        'R2': r2_score(y, predictions)
    }

    # Print metrics
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")

    # Save models and transformers
    artifacts = {
        'model.pkl': model,
        'scaler.pkl': scaler
    }
    if 'ohe' in locals():
        artifacts['ohe.pkl'] = ohe

    for filename, artifact in artifacts.items():
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(artifact, f)

    # Generate predictions for visualization
    df['Predicted_Days_to_Expiry'] = model.predict(X_scaled)
    df['Prediction_Error'] = df['Days_to_Expiry'] - df['Predicted_Days_to_Expiry']

    # Save the predictions to a CSV file
    df[['Order Date', 'Expiry Date', 'Predicted_Days_to_Expiry', 'Prediction_Error']].to_csv(
        os.path.join(output_dir, "expiry_waste_predictions.csv"), index=False)

    # Generate and save the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Order Date'], df['Prediction_Error'], alpha=0.5, color='blue', label='Prediction Error')
    plt.xlabel('Order Date')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{selected_medicine}_expiry_waste.png"))
    plt.close()

    print(f"All visualizations saved in {output_dir}.")

except Exception as e:
    print(f"Error: {str(e)}")
    raise