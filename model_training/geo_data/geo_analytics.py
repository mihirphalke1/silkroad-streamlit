import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the dataset
data = pd.read_csv('data/feature_engineered_data.csv')

# Load the location database
location_db = pd.read_csv('model_training/geo_data/city_data.csv')

# Merge the dataset with the location database for Retailer Location
data = pd.merge(
    data,
    location_db.rename(columns={"Location": "Retailer Location", "Latitude": "Retailer Latitude", "Longitude": "Retailer Longitude"}),
    on="Retailer Location",
    how="left"
)

# Merge the dataset with the location database for Distributor Location
data = pd.merge(
    data, 
    location_db.rename(columns={"Location": "Distributor Location", "Latitude": "Distributor Latitude", "Longitude": "Distributor Longitude"}),
    on="Distributor Location",
    how="left"
)

# Drop rows with missing coordinates
data.dropna(subset=['Retailer Latitude', 'Retailer Longitude', 'Distributor Latitude', 'Distributor Longitude'], inplace=True)

# Prepare data for the heatmap
heat_data = []

for _, row in data.iterrows():
    heat_data.append([row['Retailer Latitude'], row['Retailer Longitude'], row['Quantity of Order']])
    heat_data.append([row['Distributor Latitude'], row['Distributor Longitude'], row['Quantity of Order']])

# Create the map centered on India
map_center = [20.5937, 78.9629]  # Latitude and Longitude of India
m = folium.Map(location=map_center, zoom_start=5)

# Add the heatmap layer
HeatMap(heat_data).add_to(m)

# Save or display the map
m.save('plots/heatmap_india.html')  # Save to an HTML file
m  # Display in Jupyter Notebook if applicable