import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic dataset
n_customers = 10000
n_companies = 10
companies = [f'Company_{i}' for i in range(n_companies)]
customers = [f'Customer_{i}' for i in range(n_customers)]

# Simulate purchase history
data = []
for customer in customers:
    company = np.random.choice(companies)
    n_purchases = np.random.randint(1, 21)
    start_date = pd.to_datetime('2022-01-01')
    purchase_dates = start_date + pd.to_timedelta(np.random.randint(0, 1095, n_purchases), unit='D')
    purchase_amounts = np.random.uniform(10, 500, n_purchases)
    for date, amount in zip(purchase_dates, purchase_amounts):
        data.append([customer, company, date, amount])

df = pd.DataFrame(data, columns=['Customer_ID', 'Company_ID', 'Purchase_Date', 'Purchase_Amount'])

# Calculate tenure and churn status
customer_summary = df.groupby('Customer_ID').agg({
    'Purchase_Date': ['min', 'max', 'count'],
    'Purchase_Amount': 'sum',
    'Company_ID': 'first'
}).reset_index()
customer_summary.columns = ['Customer_ID', 'First_Purchase', 'Last_Purchase', 'Purchase_Count', 'Total_Amount', 'Company_ID']
customer_summary['Tenure'] = (customer_summary['Last_Purchase'] - customer_summary['First_Purchase']).dt.days
customer_summary['Churned'] = np.where(customer_summary['Last_Purchase'] < pd.to_datetime('2024-12-31') - pd.Timedelta(days=180), 1, 0)

# Save dataset (optional)
df.to_csv('customer_purchase_dataset.csv', index=False)
customer_summary.to_csv('customer_summary.csv', index=False)
print("Customer Summary Preview:")
print(customer_summary.head())

# Step 2: Time-Series Forecasting (Prophet)
def forecast_revenue_prophet(customer_id, df):
    customer_data = df[df['Customer_ID'] == customer_id][['Purchase_Date', 'Purchase_Amount']]
    customer_data = customer_data.groupby('Purchase_Date').sum().reset_index()
    customer_data.columns = ['ds', 'y']
    
    if len(customer_data) < 2:  # Skip if insufficient data
        return 0
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(customer_data)
    future = model.make_future_dataframe(periods=365)  # Forecast 1 year
    forecast = model.predict(future)
    return forecast['yhat'].sum()  # Total predicted revenue

# Step 3: ARIMA Forecasting (alternative)
def forecast_revenue_arima(customer_id, df):
    customer_data = df[df['Customer_ID'] == customer_id][['Purchase_Date', 'Purchase_Amount']]
    customer_data = customer_data.groupby('Purchase_Date').sum().resample('M').sum()['Purchase_Amount']
    
    if len(customer_data) < 5:  # Skip if insufficient data
        return 0
    
    model = ARIMA(customer_data, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.forecast(steps=12)  # Forecast 12 months
    return forecast.sum()

# Step 4: Survival Analysis (Kaplan-Meier)
kmf = KaplanMeierFitter()
kmf.fit(customer_summary['Tenure'], event_observed=customer_summary['Churned'])
avg_lifetime = kmf.median_survival_time_  # Median lifetime in days

# Step 5: Predict CLV
clv_predictions = []
for customer in customer_summary['Customer_ID']:
    prophet_clv = forecast_revenue_prophet(customer, df)
    # Use ARIMA as fallback if Prophet fails
    if prophet_clv == 0:
        prophet_clv = forecast_revenue_arima(customer, df)
    # CLV = forecasted revenue * survival probability (simplified)
    clv = prophet_clv * (1 - customer_summary[customer_summary['Customer_ID'] == customer]['Churned'].values[0])
    clv_predictions.append([customer, clv])

clv_df = pd.DataFrame(clv_predictions, columns=['Customer_ID', 'Predicted_CLV'])
clv_df = clv_df.merge(customer_summary[['Customer_ID', 'Company_ID', 'Total_Amount']], on='Customer_ID')

# Step 6: Top 50 customers by CLV
top_50_clv = clv_df.sort_values(by='Predicted_CLV', ascending=False).head(50)

print("\nTop 50 Customers by Predicted CLV:")
print(top_50_clv)

# Save results
top_50_clv.to_csv('top_50_clv_customers.csv', index=False)

# Step 7: Visualization
plt.figure(figsize=(12, 6))
top_50_clv.plot(kind='bar', x='Customer_ID', y='Predicted_CLV', legend=False)
plt.title('Top 50 Customers by Predicted CLV')
plt.xlabel('Customer ID')
plt.ylabel('Predicted CLV ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_50_clv.png')
plt.show()

# Plot survival curve
plt.figure(figsize=(10, 6))
kmf.plot()
plt.title('Customer Survival Curve (Kaplan-Meier)')
plt.xlabel('Tenure (Days)')
plt.ylabel('Survival Probability')
plt.savefig('survival_curve.png')
plt.show()