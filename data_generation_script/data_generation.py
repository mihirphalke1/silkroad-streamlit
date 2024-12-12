import pandas as pd
import numpy as np
import random
import datetime as dt
import os

# Medicine and location details
medicines = ["Paracetamol", "Amoxicillin", "Ciprofloxacin", "Ibuprofen", "Azithromycin", 
             "Metformin", "Omeprazole", "Cetirizine", "Dolo 650", "Cefixime", 
             "Aspirin", "Pantoprazole", "Levofloxacin", "Montelukast", "Atorvastatin"]

locations = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", 
             "Bangalore", "Ahmedabad", "Pune", "Jaipur", "Lucknow"]

current_date = dt.date.today()

# Assign a fixed Medicine ID to each unique medicine
medicine_ids = {med: f"M-{1000 + idx}" for idx, med in enumerate(medicines)}

# Define seasonal multipliers for each medicine (higher demand in specific months)
seasonal_factors = {
    "Paracetamol": {"12": 2.0, "1": 2.0, "2": 1.5},
    "Amoxicillin": {"6": 1.3, "7": 1.2},
    "Ciprofloxacin": {"3": 1.1},
    "Ibuprofen": {"6": 1.4, "7": 1.3},
    "Metformin": {"3": 1.1, "4": 1.05},
    "Omeprazole": {"5": 1.2},
    "Cetirizine": {"6": 1.2},
    "Dolo 650": {"12": 1.5},
    "Cefixime": {"6": 1.2},
    "Aspirin": {"10": 1.2},
    "Pantoprazole": {"1": 1.1},
    "Levofloxacin": {"12": 1.5},
    "Montelukast": {"4": 1.2},
    "Atorvastatin": {"8": 1.1},
}

# Concise descriptions for each medicine
descriptions = {
    "Paracetamol": "Relieves fever & mild to moderate pain.",
    "Amoxicillin": "Antibiotic for bacterial infections.",
    "Ciprofloxacin": "Treats bacterial infections, especially UTI.",
    "Ibuprofen": "NSAID for pain, fever, and inflammation.",
    "Azithromycin": "Antibiotic for bacterial infections.",
    "Metformin": "Controls blood sugar in type 2 diabetes.",
    "Omeprazole": "Treats GERD and other stomach issues.",
    "Cetirizine": "Antihistamine for allergy symptoms.",
    "Dolo 650": "Brand of paracetamol for pain & fever.",
    "Cefixime": "Antibiotic for respiratory and UTI infections.",
    "Aspirin": "Reduces pain, fever, inflammation, and prevents heart attacks.",
    "Pantoprazole": "Treats stomach & esophagus problems.",
    "Levofloxacin": "Antibiotic for bacterial infections.",
    "Montelukast": "Prevents asthma & seasonal allergies.",
    "Atorvastatin": "Lowers cholesterol and reduces heart disease risk.",
}

# Function to generate synthetic data with seasonality over 5 years
def generate_synthetic_data(num_records=5000):
    data = []
    start_date = current_date - dt.timedelta(days=365 * 5)  # Start date: 5 years ago
    batch_dict = {}  # To store unique batch numbers for medicine-expiry combinations

    for _ in range(num_records):
        # Select a random medicine
        name = random.choice(medicines)
        med_id = medicine_ids[name]
        order_id = f"O-{random.randint(10000, 99999)}"
        description = descriptions[name]

        # Random order date within the last 5 years
        order_date = start_date + dt.timedelta(days=random.randint(0, 365 * 5))
        expiry_date = order_date + dt.timedelta(days=random.randint(180, 1095))  # 6 months to 3 years

        # Check if the medicine-expiry combination already exists in the dictionary
        batch_key = (name, expiry_date)
        if batch_key in batch_dict:
            batch_number = batch_dict[batch_key]
        else:
            batch_number = f"B-{random.randint(1000, 9999)}"
            batch_dict[batch_key] = batch_number  # Save the batch number for this combination

        # Calculate quantity and apply seasonal demand multiplier
        quantity = random.randint(1000, 99999) // 10 * 10
        order_month = str(order_date.month)
        demand_multiplier = seasonal_factors.get(name, {}).get(order_month, 1.0)
        quantity = int(quantity * demand_multiplier)

        # Define supply chain attributes
        current_stage = random.choice(["RMS", "Man", "Dis", "Ret"])
        days_in_stage = random.randint(1, 30) if current_stage != "RMS" else (current_date - order_date).days
        days_to_expiry = (expiry_date - current_date).days if current_stage != "RMS" else None

        # Location details
        ret_location = random.choice(locations)
        dis_location = random.choice(locations)
        man_location = random.choice(locations)
        rsm_location = random.choice(locations)
        current_stage_location = {
            "RMS": rsm_location,
            "Man": man_location,
            "Dis": dis_location,
            "Ret": ret_location,
        }[current_stage]

        # Stock and sales details
        stock_at_distributor = random.randint(500, 50000) if current_stage == "Dis" else 0
        stock_status = None
        if current_stage == "Dis":
            understock_threshold = 100000
            overstock_threshold = 300000
            total_stock = stock_at_distributor + quantity
            if total_stock < understock_threshold:
                stock_status = "Understock"
            elif total_stock > overstock_threshold:
                stock_status = "Overstock"
            else:
                stock_status = "Optimal"

        sales_quantity = None
        sale_date = None
        if current_stage == "Ret":
            sales_quantity = random.randint(50, 1000) * 10

        # Calculate sale date logically
        days_diff = (current_date - order_date).days
        if days_diff > 0:
            sale_date = order_date + dt.timedelta(days=random.randint(1, days_diff))
        else:
            sale_date = order_date  # Default to order_date if no valid range

        # Append data row
        data.append([
            med_id, order_id, name, description, batch_number, order_date, quantity, 
            current_stage, current_stage_location, days_in_stage, expiry_date, days_to_expiry, 
            ret_location, dis_location, man_location, rsm_location, stock_at_distributor, 
            stock_status, sales_quantity, sale_date
        ])

    return pd.DataFrame(data, columns=[
        "Medicine ID", "Order ID", "Name", "Description", "Batch Number", "Order Date", 
        "Quantity of Order", "Current Stage", "Current Stage Location", "Days in Current Stage", 
        "Expiry Date", "Days to Expiry", "Retailer Location", "Distributor Location", 
        "Manufacturer Location", "Raw Material Supplier Location", "Stock at Distributor", 
        "Stock Status", "Sales Quantity", "Sale Date"
    ])

if __name__ == "__main__":
    num_records = 1000  # Adjust as needed
    synthetic_data = generate_synthetic_data(num_records)
    os.makedirs('data', exist_ok=True)
    synthetic_data.to_csv('data/synthetic_data.csv', index=False)
    print("Synthetic data for 5 years generated and saved to 'data/synthetic_data_5_years.csv'")