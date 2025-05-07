import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
import plotly.express as px
from flask import Flask, request, jsonify
import uuid
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('vader_lexicon')

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Market Research
# Understand factors driving ad engagement: positive sentiment, readable copy, vibrant visuals, and hotel appeal.
print("Market Research: Identified key features - ad copy sentiment, readability, visual appeal, hotel location, star rating.")

# Step 2: Data Collection
# Generate synthetic dataset for 9,000 hotel ads.
n_ads = 9000
locations = ['Goa', 'Mumbai', 'Delhi', 'Jaipur', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Kerala', 'Udaipur']
ad_copy_templates = [
    "Discover {adj} stays at our {star}-star hotel in {location}!",
    "Book now for {adj} deals at {location}'s best hotel!",
    "Experience {adj} luxury at our {star}-star {location} hotel!",
    "Unforgettable {adj} vacations start at our {location} hotel!",
    "Stay in {adj} comfort at our {star}-star hotel in {location}!"
]
adjectives = ['luxury', 'affordable', 'unforgettable', 'relaxing', 'exclusive', 'budget-friendly', 'stunning']

data = {
    'ad_id': [str(uuid.uuid4()) for _ in range(n_ads)],
    'hotel_name': [f"Hotel_{i}" for i in range(1, n_ads + 1)],
    'location': np.random.choice(locations, n_ads),
    'star_rating': np.random.choice([3, 4, 5], n_ads, p=[0.3, 0.5, 0.2]),
    'price_per_night': np.random.uniform(2000, 20000, n_ads).round(2),  # INR
    'ad_copy': [
        np.random.choice(ad_copy_templates).format(
            adj=np.random.choice(adjectives),
            star=np.random.choice([3, 4, 5]),
            location=np.random.choice(locations)
        ) for _ in range(n_ads)
    ],
    'visual_appeal_score': np.random.uniform(0.5, 1.0, n_ads).round(2),  # 0.5–1.0
    'color_vibrancy': np.random.uniform(0.5, 1.0, n_ads).round(2),  # 0.5–1.0
    'image_quality': np.random.uniform(0.5, 1.0, n_ads).round(2),  # 0.5–1.0
    'likes': np.zeros(n_ads),
    'shares': np.zeros(n_ads),
    'comments': np.zeros(n_ads)
}

df = pd.DataFrame(data)

# Generate sentiment and readability scores
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['ad_copy'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['readability_score'] = df['ad_copy'].apply(flesch_reading_ease)

# Generate synthetic engagement metrics
for i, row in df.iterrows():
    base_engagement = 100 * row['sentiment_score'] + 50 * row['visual_appeal_score']
    star_effect = {3: 0.8, 4: 1.0, 5: 1.2}[row['star_rating']] * 50
    price_effect = -0.01 * row['price_per_night'] + 100
    readability_effect = 0.5 * row['readability_score']
    visual_effect = 30 * (row['color_vibrancy'] + row['image_quality'])
    total_engagement = base_engagement + star_effect + price_effect + readability_effect + visual_effect + np.random.normal(0, 20)
    total_engagement = max(0, int(total_engagement))
    # Distribute engagement across likes, shares, comments
    df.at[i, 'likes'] = int(total_engagement * np.random.uniform(0.5, 0.7))
    df.at[i, 'shares'] = int(total_engagement * np.random.uniform(0.1, 0.3))
    df.at[i, 'comments'] = int(total_engagement * np.random.uniform(0.1, 0.2))

# Combine engagement metrics
df['total_engagement'] = df['likes'] + df['shares'] + df['comments']

# Save synthetic dataset
df.to_csv('hotel_ads_synthetic.csv', index=False)
print("Data Collection: Generated synthetic dataset for 9,000 hotel ads.")

# Step 3: Data Cleaning
# Handle missing values and outliers.
print("Data Cleaning: Checking for missing values...")
print(df.isnull().sum())

# Cap total_engagement at 95th percentile
engagement_cap = df['total_engagement'].quantile(0.95)
df['total_engagement'] = df['total_engagement'].clip(upper=engagement_cap)
df['likes'] = df['likes'].clip(upper=df['total_engagement'] * 0.7)
df['shares'] = df['shares'].clip(upper=df['total_engagement'] * 0.3)
df['comments'] = df['comments'].clip(upper=df['total_engagement'] * 0.2)

# Ensure positive values
df['price_per_night'] = df['price_per_night'].clip(lower=0)
df[['likes', 'shares', 'comments', 'total_engagement']] = df[['likes', 'shares', 'comments', 'total_engagement']].clip(lower=0)
print("Data Cleaning: Handled outliers and ensured positive values.")

# Step 4: Exploratory Data Analysis (EDA)
# Analyze distributions.
print("EDA: Summary statistics...")
print(df.describe())

# Visualize total_engagement by star_rating
fig_eda = px.box(
    df,
    x='star_rating',
    y='total_engagement',
    title='Total Engagement by Star Rating',
    labels={'star_rating': 'Star Rating', 'total_engagement': 'Total Engagement (Likes + Shares + Comments)'}
)
fig_eda.update_layout(template='plotly_white')
fig_eda.write_html('eda_engagement_by_star.html')
print("EDA: Generated box plot of engagement by star rating.")

# Step 5: Feature Engineering
# Encode categorical features and select features for modeling.
le_location = LabelEncoder()
df['location_encoded'] = le_location.fit_transform(df['location'])

# Features for regression
features = [
    'location_encoded', 'star_rating', 'price_per_night', 'sentiment_score',
    'readability_score', 'visual_appeal_score', 'color_vibrancy', 'image_quality'
]

print("Feature Engineering: Encoded location and computed sentiment/readability scores.")

# Step 6: Model Selection
# Choose Random Forest Regressor for non-linear relationships.
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Model Selection: Chose Random Forest Regressor.")

# Step 7: Model Training
# Split data and train.
X = df[features]
y = df['total_engagement']
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
# Predict engagement and find top 10 hotels.
df['predicted_engagement'] = best_model.predict(X)
top_10_hotels = df.sort_values(by='predicted_engagement', ascending=False).head(10)

# Save outputs
df.to_csv('ad_predictions.csv', index=False)
top_10_hotels.to_csv('top_10_hotels.csv', index=False)
print("\nTop 10 Hotels with Highest Predicted Engagement:")
print(top_10_hotels[['ad_id', 'hotel_name', 'location', 'star_rating', 'likes', 'shares', 'comments', 'predicted_engagement']])

# Step 11: Visualization
# Scatter plot: Predicted vs. actual engagement
fig_pred = px.scatter(
    x=y_test, y=y_pred_best,
    title='Predicted vs. Actual Total Engagement',
    labels={'x': 'Actual Engagement', 'y': 'Predicted Engagement'}
)
fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='red', dash='dash'))
fig_pred.update_layout(template='plotly_white')
fig_pred.write_html('predicted_vs_actual_engagement.html')

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
# Simulate Flask API for predicting engagement.
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df['location_encoded'] = le_location.transform(input_df['location'])
    input_df['sentiment_score'] = input_df['ad_copy'].apply(lambda x: sia.polarity_scores(x)['compound'])
    input_df['readability_score'] = input_df['ad_copy'].apply(flesch_reading_ease)
    input_features = input_df[features]
    prediction = best_model.predict(input_features)[0]
    return jsonify({'predicted_engagement': round(prediction, 2)})

# Simulate running Flask app (commented out)
# if __name__ == '__main__':
#     app.run(debug=True)
print("Deployment: Simulated Flask API for predicting ad engagement.")

# Step 13: Monitoring and Maintenance
# Plan to monitor and update model.
print("Monitoring and Maintenance: Monitor engagement predictions, collect new ad data, and retrain periodically.")