import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Create Synthetic Dataset for Energy Prices
np.random.seed(42)
n_days = 90  # 90 days of historical data
start_date = datetime(2025, 1, 4)  # Start 90 days before today
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Features affecting prices
demand = np.random.normal(5000, 500, n_days)  # MW for electricity, MMBtu for gas
supply = np.random.normal(5500, 600, n_days)  # MW/MMBtu
temp = np.random.normal(15, 5, n_days)  # °C

# Electricity prices ($/MWh): Higher with high demand, low supply, extreme temps
elec_price = 50 + 0.01 * (demand - supply) + 0.5 * abs(temp - 15) + np.random.normal(0, 5, n_days)
elec_price = np.clip(elec_price, 20, 100)  # Realistic range
elec_label = (elec_price > 60).astype(int)  # High (1) if > $60/MWh

# Gas prices ($/MMBtu): Similar logic
gas_price = 3 + 0.002 * (demand - supply) + 0.1 * abs(temp - 15) + np.random.normal(0, 0.5, n_days)
gas_price = np.clip(gas_price, 2, 10)  # Realistic range
gas_label = (gas_price > 5).astype(int)  # High (1) if > $5/MMBtu

df = pd.DataFrame({
    "Date": dates,
    "Demand": demand,
    "Supply": supply,
    "Temperature": temp,
    "Electricity_Price": elec_price,
    "Electricity_Label": elec_label,
    "Gas_Price": gas_price,
    "Gas_Label": gas_label
})

print(f"Dataset shape: {df.shape}")
print(df.head())

# Save dataset to CSV
output_dir = "C:/Users/HP/OneDrive/Documents/Green Energy Models"  # Adjust path as needed
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/energy_price_data.csv", index=False)
print(f"Dataset saved to {output_dir}/energy_price_data.csv")

# Step 2: Prepare Data for Random Forest Classifier
features = ["Demand", "Supply", "Temperature"]
X = df[features].values
y_elec = df["Electricity_Label"].values
y_gas = df["Gas_Label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train Random Forest Classifier
elec_rf = RandomForestClassifier(n_estimators=100, random_state=42)
elec_rf.fit(X_scaled, y_elec)

gas_rf = RandomForestClassifier(n_estimators=100, random_state=42)
gas_rf.fit(X_scaled, y_gas)

# Evaluate models
elec_score = elec_rf.score(X_scaled, y_elec)
gas_score = gas_rf.score(X_scaled, y_gas)
print(f"Electricity Price Training Accuracy: {elec_score:.2f}")
print(f"Gas Price Training Accuracy: {gas_score:.2f}")

# Step 4: Forecast Prices for Next 10 Days (April 4-13, 2025)
forecast_dates = [datetime(2025, 4, 4) + timedelta(days=i) for i in range(10)]
forecast_demand = np.random.normal(5000, 500, 10)  # Simulated future data
forecast_supply = np.random.normal(5500, 600, 10)
forecast_temp = np.random.normal(20, 5, 10)  # April avg ~20°C

forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Demand": forecast_demand,
    "Supply": forecast_supply,
    "Temperature": forecast_temp
})

X_forecast = forecast_df[features].values
X_forecast_scaled = scaler.transform(X_forecast)

elec_probs = elec_rf.predict_proba(X_forecast_scaled)[:, 1]  # Probability of High
gas_probs = gas_rf.predict_proba(X_forecast_scaled)[:, 1]

forecast_df["Electricity_Prob_High"] = elec_probs
forecast_df["Gas_Prob_High"] = gas_probs
forecast_df["Electricity_Price"] = 50 + (elec_probs * 50)  # Scale to $50-$100/MWh
forecast_df["Gas_Price"] = 3 + (gas_probs * 7)              # Scale to $3-$10/MMBtu

# Save forecast to CSV
forecast_df.to_csv(f"{output_dir}/energy_price_forecast.csv", index=False)
print(f"Forecast saved to {output_dir}/energy_price_forecast.csv")

# Step 5: Visualizations
sns.set_style("darkgrid")

# Line Graph: Historical and Forecasted Prices
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Electricity_Price"], label="Historical Electricity ($/MWh)", lw=2, color="blue", alpha=0.5)
plt.plot(forecast_df["Date"], forecast_df["Electricity_Price"], label="Forecasted Electricity ($/MWh)", lw=2, color="blue")
plt.plot(df["Date"], df["Gas_Price"], label="Historical Gas ($/MMBtu)", lw=2, color="orange", alpha=0.5)
plt.plot(forecast_df["Date"], forecast_df["Gas_Price"], label="Forecasted Gas ($/MMBtu)", lw=2, color="orange")
plt.axvline(x=datetime(2025, 4, 3), color="gray", linestyle="--", label="Forecast Start")
plt.title("Energy Price Forecast (Jan 4 - Apr 13, 2025)", fontsize=14, weight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Chart: Probability of High Prices
plt.figure(figsize=(10, 6))
bar_data = pd.DataFrame({
    "Date": forecast_dates,
    "Electricity": elec_probs,
    "Gas": gas_probs
})
bar_data = bar_data.melt(id_vars=["Date"], var_name="Energy_Type", value_name="Prob_High")
sns.barplot(x="Date", y="Prob_High", hue="Energy_Type", data=bar_data, palette="Set2")
plt.title("Probability of High Energy Prices (Apr 4-13, 2025)", fontsize=14, weight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Probability of High Price", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Energy Type")
plt.tight_layout()
plt.show()

# Print Forecast Summary
print("\nEnergy Price Forecast Summary (Apr 4-13, 2025):")
print(forecast_df[["Date", "Electricity_Price", "Gas_Price"]])