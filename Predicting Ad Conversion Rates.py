import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from flask import Flask, request, jsonify
import uuid
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Market Research
# Understand Facebook Ads and hotel booking domain: key factors include ad budget, audience targeting, creative quality, and location.
print("Market Research: Identified key features - location, ad budget, audience size, ad creative score, click-through rate.")

# Step 2: Data Collection
# Generate synthetic dataset for 1,000 hotels with ad-related features.
n_hotels = 1000
locations = ['Goa', 'Mumbai', 'Delhi', 'Jaipur', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Kerala', 'Udaipur']
hotel_names = [f"Hotel_{i}" for i in range(1, n_hotels + 1)]
hotel_ids = [str(uuid.uuid4()) for _ in range(n_hotels)]

data = {
    'hotel_id': hotel_ids,
    'hotel_name': hotel_names,
    'location': np.random.choice(locations, n_hotels),
    'star_rating': np.random.choice([3, 4, 5], n_hotels, p=[0.3, 0.5, 0.2]),
    'ad_budget': np.random.uniform(5000, 50000, n_hotels).round(2),  # INR
    'audience_size': np.random.randint(10000, 1000000, n_hotels),
    'ad_creative_score': np.random.uniform(0.5, 1.0, n_hotels).round(2),  # 0.5–1.0
    'click_through_rate': np.random.uniform(0.01, 0.1, n_hotels).round(3),  # 1–10%
    'seasonality_index': np.random.uniform(0.8, 1.4, n_hotels).round(2),  # 0.8–1.4
    'conversions': np.zeros(n_hotels)  # Target: bookings from ads
}

df = pd.DataFrame(data)

# Generate synthetic conversions
for i, row in df.iterrows():
    base_conversions = row['audience_size'] * row['click_through_rate'] * 0.05  # Base conversion rate
    creative_effect = row['ad_creative_score'] * 20
    budget_effect = row['ad_budget'] * 0.0001
    star_multiplier = {3: 0.8, 4: 1.0, 5: 1.2}[row['star_rating']]
    seasonality_effect = row['seasonality_index'] * 10
    conversions = base_conversions * star_multiplier + creative_effect + budget_effect + seasonality_effect + np.random.normal(0, 5)
    df.at[i, 'conversions'] = max(0, int(conversions))

# Save synthetic dataset
df.to_csv('fb_ads_hotel_synthetic.csv', index=False)
print("Data Collection: Generated synthetic dataset for 1,000 hotels with ad features.")

# Step 3: Data Cleaning
# Handle missing values, outliers, and inconsistencies.
print("Data Cleaning: Checking for missing values...")
print(df.isnull().sum())

# Cap conversions at 95th percentile to handle outliers
conversion_cap = df['conversions'].quantile(0.95)
df['conversions'] = df['conversions'].clip(upper=conversion_cap)

# Ensure ad_budget and audience_size are positive
df['ad_budget'] = df['ad_budget'].clip(lower=0)
df['audience_size'] = df['audience_size'].clip(lower=0)
print("Data Cleaning: Handled outliers and ensured positive values.")

# Step 4: Exploratory Data Analysis (EDA)
# Analyze distributions and relationships.
print("EDA: Summary statistics...")
print(df.describe())

# Visualize conversions by location
fig_eda = px.box(
    df,
    x='location',
    y='conversions',
    title='Ad Conversions by Location',
    labels={'location': 'Location', 'conversions': 'Conversions'}
)
fig_eda.update_layout(template='plotly_white')
fig_eda.write_html('eda_conversions_by_location.html')
print("EDA: Generated box plot of conversions by location.")

# Step 5: Feature Engineering
# Create or transform features.
le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])

# Create budget per audience feature
df['budget_per_audience'] = df['ad_budget'] / df['audience_size']

# Log-transform audience_size
df['log_audience_size'] = np.log1p(df['audience_size'])

features = [
    'location_encoded', 'star_rating', 'ad_budget', 'log_audience_size',
    'ad_creative_score', 'click_through_rate', 'seasonality_index', 'budget_per_audience'
]
print("Feature Engineering: Created encoded location, budget per audience, and log-transformed audience size.")

# Step 6: Model Selection
# Choose Random Forest Regressor for non-linear relationships.
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Model Selection: Chose Random Forest Regressor.")

# Step 7: Model Training
# Split data and train.
X = df[features]
y = df['conversions']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Model Training: Trained Random Forest Regressor.")

# Step 8: Model Evaluation
# Evaluate with MAE, RMSE, R².
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation: Performance metrics...")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Model Tuning
# Grid search for hyperparameter optimization.
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Model Tuning: Best parameters:", grid_search.best_params_)

# Re-evaluate with best model
y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print("Model Evaluation (Tuned): Performance metrics...")
print(f"Mean Absolute Error (MAE): {mae_best:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_best:.2f}")
print(f"R² Score: {r2_best:.2f}")

# Step 10: Prediction and Insights
# Predict conversions and find top 5 hotels.
df['predicted_conversions'] = best_model.predict(X)
top_5_hotels = df.sort_values(by='predicted_conversions', ascending=False).head(5)

# Save top 5 hotels
top_5_hotels.to_csv('top_5_hotels_ads.csv', index=False)
print("\nTop 5 Hotels with Highest Predicted Ad Conversions:")
print(top_5_hotels[['hotel_id', 'hotel_name', 'location', 'star_rating', 'predicted_conversions']])

# Step 11: Visualization
# Scatter plot: Predicted vs. actual conversions
fig_pred = px.scatter(
    x=y_test, y=y_pred_best,
    title='Predicted vs. Actual Ad Conversions',
    labels={'x': 'Actual Conversions', 'y': 'Predicted Conversions'}
)
fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='red', dash='dash'))
fig_pred.update_layout(template='plotly_white')
fig_pred.write_html('predicted_vs_actual_conversions.html')

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

fig_importance = px.bar(
    feature_importance,
    x='importance',
    y='feature',
    title='Feature Importance in Random Forest Regressor',
    labels={'importance': 'Importance', 'feature': 'Feature'},
    orientation='h'
)
fig_importance.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})
fig_importance.write_html('feature_importance.html')
print("Visualization: Generated scatter and feature importance plots.")

# Step 12: Deployment
# Simulate Flask API for predicting ad conversions.
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df['location_encoded'] = le.transform(input_df['location'])
    input_df['log_audience_size'] = np.log1p(input_df['audience_size'])
    input_df['budget_per_audience'] = input_df['ad_budget'] / input_df['audience_size']
    input_features = input_df[features]
    prediction = best_model.predict(input_features)[0]
    return jsonify({'predicted_conversions': round(prediction, 2)})

# Simulate running Flask app (commented out)
# if __name__ == '__main__':
#     app.run(debug=True)
print("Deployment: Simulated Flask API for predicting ad conversions.")

# Step 13: Monitoring and Maintenance
# Plan to monitor and retrain model.
print("Monitoring and Maintenance: Monitor predictions, collect new ad data, and retrain periodically.")