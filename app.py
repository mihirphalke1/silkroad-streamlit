import streamlit as st
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load your data
@st.cache_data
def load_data():
    return pd.read_csv("data/feature_engineered_data.csv", parse_dates=["Order Date"])

# Sidebar for selecting the medicine
def sidebar():
    st.sidebar.title("Medicine Selection")
    data = load_data()
    medicine_names = sorted(data['Name'].unique())
    selected_medicine = st.sidebar.selectbox("Select Medicine", medicine_names)
    return selected_medicine, data

# Display saved plots from directory
def display_saved_plots(medicine, plot_type):
    plot_dirs = {
        "historical_demand": "plots/historical_demand",
        "future_demand": "plots/future_demand",
        "spike_analysis": "plots/spike",
        "expiry_waste": "outputs_expiry"
    }
    
    # Check if the plot type exists in the dictionary
    if plot_type not in plot_dirs:
        st.error(f"Plot type '{plot_type}' not found in the directory.")
        return False

    plot_path = os.path.join(plot_dirs[plot_type], f"{medicine}_{plot_type}.png")
    if os.path.exists(plot_path):
        st.image(plot_path, caption=f"{plot_type.replace('_', ' ').title()} for {medicine}", use_column_width=True)
        return True
    else:
        st.warning(f"No plot found for {plot_type.replace('_', ' ').title()} for {medicine}. Please ensure the plots are generated.")
        return False

# Display corresponding table
def display_scrollable_table(feature_data):
    if feature_data is not None and not feature_data.empty:
        st.subheader("Relevant Data")
        st.dataframe(feature_data, use_container_width=True)
    else:
        st.warning("No relevant data available for display.")

# Main Layout
def main():
    # Set page configuration first
    st.set_page_config(page_title="Drug Supply Chain Analysis and Predictions", layout="wide")

    # Sidebar to select medicine
    selected_medicine, data = sidebar()
    
    # Filter data for selected medicine
    medicine_data = data[data['Name'] == selected_medicine]

    # Page Title
    st.title(f"Drug Supply Chain Analysis and Predictions {selected_medicine}")

    # Display Historical Demand plot and its corresponding data
    st.subheader("1. Historical Demand")
    if display_saved_plots(selected_medicine, "historical_demand"):
        display_scrollable_table(medicine_data[['Order Date', 'Quantity of Order']])

    # Display Demand Forecast plot and its corresponding data
    st.subheader("2. Demand Forecast")
    if display_saved_plots(selected_medicine, "future_demand"):
        months = pd.date_range("2024-07-01", "2024-12-01", freq="M")  # 6 months
        forecast = np.random.rand(len(months)) * 1000  # Generate random forecast data for the correct number of months
        forecast_data = pd.DataFrame({
            "Month": months,
            "Forecasted Demand": forecast
        })
        display_scrollable_table(forecast_data)

    # Display Spike Analysis plot and its corresponding data
    st.subheader("3. Demand Spike Analysis")
    if display_saved_plots(selected_medicine, "spike_analysis"):
        spike_analysis_data = medicine_data.groupby(medicine_data['Order Date'].dt.to_period('M'))['Quantity of Order'].sum().diff()
        spike_df = spike_analysis_data.reset_index().rename(columns={'Order Date': 'Month', 'Quantity of Order': 'Spike Change'})
        display_scrollable_table(spike_df)

    st.subheader("4. Key Insights")
    
    # Calculate metrics
    total_orders = medicine_data['Quantity of Order'].sum()
    avg_monthly_demand = medicine_data.groupby(medicine_data['Order Date'].dt.to_period('M'))['Quantity of Order'].sum().mean()
    max_monthly_demand = medicine_data.groupby(medicine_data['Order Date'].dt.to_period('M'))['Quantity of Order'].sum().max()
    
    # Display metrics
    st.metric("Total Quantity Ordered until now", f"{total_orders:,.0f}")
    st.metric("Average Monthly Demand", f"{avg_monthly_demand:,.0f}")
    st.metric("Peak Monthly Demand", f"{max_monthly_demand:,.0f}")


    # Footer
    st.markdown("---")
    st.markdown("**Web3 Supply Chain Management System - Medicine Analytics Dashboard**")

if __name__ == "__main__":
    main()