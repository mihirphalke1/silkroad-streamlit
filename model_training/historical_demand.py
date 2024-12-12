import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
data = pd.read_csv("data/feature_engineered_data.csv", parse_dates=["Order Date"])

# Prepare the data: Aggregate by medicine and month
data["Order Month"] = data["Order Date"].dt.to_period("M")
grouped_data = data.groupby(["Name", "Order Month"])["Quantity of Order"].sum().reset_index()
grouped_data["Order Month"] = grouped_data["Order Month"].dt.to_timestamp()

# Filter the data to include only up to June 2024
end_date = pd.Timestamp("2024-06-30")
grouped_data = grouped_data[grouped_data["Order Month"] <= end_date]

# Create output directory for individual plots
output_dir = "plots/historical_demand"
os.makedirs(output_dir, exist_ok=True)

# Colors for each medicine
colors = plt.cm.tab20.colors
color_idx = 0

# Loop through each medicine for plotting historical data
for medicine in grouped_data["Name"].unique():
    # Filter data for the specific medicine
    med_data = grouped_data[grouped_data["Name"] == medicine]
    
    # Initialize the plot
    plt.figure(figsize=(16, 10))
    plt.plot(med_data["Order Month"], med_data["Quantity of Order"], label=f"{medicine} (Historical)", color=colors[color_idx], linestyle="--")
    
    # Customize the plot
    plt.title(f"Historical Demand for {medicine} (Up to June 2024)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Demand (Quantity of Order)", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{medicine}_historical_demand.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Cycle through colors
    color_idx = (color_idx + 1) % len(colors)

print(f"Historical demand plots saved in '{output_dir}' directory.")