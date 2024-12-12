import pandas as pd

# Load the existing CSV data
df = pd.read_csv("data/synthetic_data.csv")

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Generate unique IDs for RMS, Manufacturer, Distributor, and Retailer
rms_ids = {rms: f"RMS-{i+1}" for i, rms in enumerate(df['Raw Material Supplier Location'].unique())}
man_ids = {man: f"MAN-{i+1}" for i, man in enumerate(df['Manufacturer Location'].unique())}
dis_ids = {dis: f"DIS-{i+1}" for i, dis in enumerate(df['Distributor Location'].unique())}
ret_ids = {ret: f"RET-{i+1}" for i, ret in enumerate(df['Retailer Location'].unique())}

# Ensure numeric columns are of the correct type
df['Stock at Distributor'] = pd.to_numeric(df['Stock at Distributor'], errors='coerce')
df['Days to Expiry'] = pd.to_numeric(df['Days to Expiry'], errors='coerce')
df['Quantity of Order'] = pd.to_numeric(df['Quantity of Order'], errors='coerce')
df['Sales Quantity'] = pd.to_numeric(df['Sales Quantity'], errors='coerce')
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce')

# Feature Engineering
df['Order Shortfall'] = df['Quantity of Order'] - df['Stock at Distributor'].fillna(0)
df['Expiry Risk Score'] = df['Stock at Distributor'] / df['Days to Expiry'].replace(0, 1)
df['Turn Over Ratio'] = df['Sales Quantity'] / df['Quantity of Order'].replace(0, 1)
df['Supply Chain Velocity'] = (df['Sale Date'] - df['Order Date']).dt.days.fillna(0)
df['Stage Transition Time'] = df['Days in Current Stage'].fillna(0)
critical_threshold = 10000
df['Time to Critical'] = (df['Stock at Distributor'] - critical_threshold).clip(lower=0)
df['Predicted Expiry Waste'] = (df['Days to Expiry'] / df['Sales Quantity'].replace(0, 1)) * df['Quantity of Order'].fillna(0)
df['Anomaly Score'] = (df['Stock at Distributor'] - df['Quantity of Order']).abs()

# Correct Lead Time Calculation
df['Lead Time'] = (df['Sale Date'] - df['Order Date']).dt.days
df['Lead Time'] = df['Lead Time'].fillna(0).clip(lower=0)  # Replace missing values with 0 and ensure non-negative values

# Map the IDs to their respective columns
df['RMS ID'] = df['Raw Material Supplier Location'].map(rms_ids)
df['Manufacturer ID'] = df['Manufacturer Location'].map(man_ids)
df['Distributor ID'] = df['Distributor Location'].map(dis_ids)
df['Retailer ID'] = df['Retailer Location'].map(ret_ids)

# Reorder columns to place IDs next to their respective location columns
column_order = []
for col in df.columns:
    column_order.append(col)
    if col == "Raw Material Supplier Location":
        column_order.append("RMS ID")
    elif col == "Manufacturer Location":
        column_order.append("Manufacturer ID")
    elif col == "Distributor Location":
        column_order.append("Distributor ID")
    elif col == "Retailer Location":
        column_order.append("Retailer ID")

# Reorder the DataFrame columns
df = df[[col for col in column_order if col in df.columns]]

# Save the updated DataFrame to a new CSV
df.fillna(0, inplace=True)
df.to_csv('data/feature_engineered_data_with_ids.csv', index=False)

print("Feature engineering and ID assignment completed with corrected Lead Time.")