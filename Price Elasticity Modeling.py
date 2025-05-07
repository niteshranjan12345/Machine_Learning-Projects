import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic dataset
n_products = 10000
products = [
    'rice', 'wheat', 'milk', 'bread', 'butter', 'eggs', 'cheese', 
    'yogurt', 'tea', 'coffee', 'sugar', 'salt', 'oil', 'dal', 'flour'
]
categories = ['grains', 'dairy', 'beverages', 'spices', 'oils']

data = {
    'Product_ID': range(1, n_products + 1),
    'Product_Name': np.random.choice(products, n_products),
    'Price': np.random.uniform(10, 500, n_products),  # ₹
    'Category': np.random.choice(categories, n_products),
    'Competitor_Price': np.random.uniform(10, 500, n_products)
}

df = pd.DataFrame(data)

# Simulate demand (linear relationship with price + noise)
df['Demand'] = (
    1000 - 2 * df['Price'] +  # Base demand decreases with price
    0.5 * (df['Competitor_Price'] - df['Price']) +  # Competitor effect
    np.random.normal(0, 50, n_products)  # Noise
).clip(0)  # Ensure demand >= 0

# Save dataset (optional)
df.to_csv('daily_eating_products_dataset.csv', index=False)
print("Dataset Preview:")
print(df.head())

# Step 2: Linear Regression for Price Elasticity
X = df[['Price', 'Competitor_Price']]  # Features
y = df['Demand']  # Target

model = LinearRegression()
model.fit(X, y)

# Coefficients
price_coeff = model.coef_[0]  # Price coefficient
print(f"\nPrice Coefficient: {price_coeff:.4f}")

# Calculate elasticity (at mean price and demand)
mean_price = df['Price'].mean()
mean_demand = df['Demand'].mean()
elasticity = price_coeff * (mean_price / mean_demand)
print(f"Price Elasticity of Demand: {elasticity:.4f}")

# Step 3: Optimize pricing for maximum revenue
# Revenue = Price * Demand
# Demand = intercept + price_coeff * Price + competitor_coeff * Competitor_Price
intercept = model.intercept_
competitor_coeff = model.coef_[1]

# Optimal price: maximize Revenue = Price * (intercept + price_coeff * Price + competitor_coeff * Competitor_Price)
# Derivative of Revenue w.r.t Price = 0 => Price_opt = -(intercept + competitor_coeff * Competitor_Price) / (2 * price_coeff)
df['Optimal_Price'] = -(intercept + competitor_coeff * df['Competitor_Price']) / (2 * price_coeff)
df['Optimal_Price'] = df['Optimal_Price'].clip(10, 500)  # Constrain within realistic range
df['Predicted_Demand'] = model.predict(X)
df['Optimal_Demand'] = intercept + price_coeff * df['Optimal_Price'] + competitor_coeff * df['Competitor_Price']
df['Current_Revenue'] = df['Price'] * df['Predicted_Demand']
df['Optimal_Revenue'] = df['Optimal_Price'] * df['Optimal_Demand']

# Step 4: Top 30 high-demand products (based on current demand)
top_30_demand = df.sort_values(by='Demand', ascending=False).head(30)
print("\nTop 30 High-Demand Products:")
print(top_30_demand[['Product_Name', 'Price', 'Demand', 'Optimal_Price', 'Optimal_Demand', 'Current_Revenue', 'Optimal_Revenue']])

# Save results
top_30_demand.to_csv('top_30_high_demand_products.csv', index=False)

# Step 5: Visualization
plt.figure(figsize=(12, 6))
plt.bar(top_30_demand['Product_Name'], top_30_demand['Price'], label='Actual Price (₹)', alpha=0.7)
plt.bar(top_30_demand['Product_Name'], top_30_demand['Optimal_Price'], label='Optimal Price (₹)', alpha=0.5)
plt.title('Actual vs Optimal Prices for Top 30 High-Demand Products')
plt.xlabel('Product Name')
plt.ylabel('Price (₹)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('price_comparison.png')
plt.show()