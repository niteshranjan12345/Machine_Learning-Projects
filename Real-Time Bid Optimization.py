import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.figure_factory as ff
from flask import Flask, request, jsonify
import uuid
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Market Research
# Understand factors driving ad performance: CTR, CVR, ad relevance, seasonality, hotel rating.
print("Market Research: Identified key features - CTR, CVR, ad relevance, seasonality, hotel rating, audience demographics.")

# Step 2: Data Collection
# Generate synthetic dataset for 35,000 hotel ad campaigns.
n_campaigns = 35000
destinations = ['Goa', 'Jaipur', 'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Kerala', 'Udaipur']
audience_age_groups = ['18-35', '36-50', '51+']
audience_income_levels = ['Low', 'Medium', 'High']
months = ['May-2025', 'Jun-2025', 'Jul-2025', 'Aug-2025', 'Sep-2025', 'Oct-2025', 'Nov-2025', 'Dec-2025', 'Jan-2026', 'Feb-2026', 'Mar-2026', 'Apr-2026']

data = {
    'campaign_id': [str(uuid.uuid4()) for _ in range(n_campaigns)],
    'ad_name': [f"{np.random.choice(destinations)}_Ad_{i}" for i in range(1, n_campaigns + 1)],
    'destination': np.random.choice(destinations, n_campaigns),
    'month': np.random.choice(months, n_campaigns),
    'bid_amount': np.random.uniform(10, 500, n_campaigns).round(2),  # INR
    'impressions': np.random.randint(1000, 100000, n_campaigns),
    'hotel_rating': np.random.choice([3, 4, 5], n_campaigns, p=[0.3, 0.5, 0.2]),
    'ad_relevance_score': np.random.uniform(0.5, 1.0, n_campaigns).round(2),  # 0.5–1.0
    'audience_age_group': np.random.choice(audience_age_groups, n_campaigns),
    'audience_income_level': np.random.choice(audience_income_levels, n_campaigns)
}

df = pd.DataFrame(data)

# Generate clicks and conversions
df['clicks'] = (df['impressions'] * np.random.uniform(0.005, 0.015, n_campaigns) * df['ad_relevance_score']).astype(int)
df['conversions'] = (df['clicks'] * np.random.uniform(0.01, 0.1, n_campaigns) * df['hotel_rating'] / 5 * df['ad_relevance_score']).astype(int)

# Ensure non-negative values
df['clicks'] = df['clicks'].clip(lower=0)
df['conversions'] = df['conversions'].clip(lower=0)

# Compute derived metrics
df['ctr'] = df['clicks'] / df['impressions']
df['cvr'] = df['conversions'] / df['clicks'].replace(0, 1)  # Avoid division by zero
df['cpc'] = df['bid_amount'] / df['clicks'].replace(0, 1)
df['cpa'] = df['bid_amount'] / df['conversions'].replace(0, 1)

# Assign season (peak: May-Nov 2025, off-peak: others)
df['season'] = df['month'].apply(lambda x: 'Peak' if x in ['May-2025', 'Jun-2025', 'Jul-2025', 'Aug-2025', 'Sep-2025', 'Oct-2025', 'Nov-2025'] else 'Off-Peak')

# Label campaigns as high-performing (1) or low-performing (0)
# High-performing: High conversions and low CPA
median_conversions = df['conversions'].median()
median_cpa = df['cpa'].median()
df['performance_label'] = ((df['conversions'] > median_conversions) & (df['cpa'] < median_cpa)).astype(int)

# Save synthetic dataset
df.to_csv('hotel_ad_campaigns.csv', index=False)
print("Data Collection: Generated synthetic dataset for 35,000 hotel ad campaigns.")

# Step 3: Data Cleaning
# Handle missing values and outliers.
print("Data Cleaning: Checking for missing values...")
print(df.isnull().sum())

# Cap extreme values
df['ctr'] = df['ctr'].clip(upper=df['ctr'].quantile(0.95))
df['cvr'] = df['cvr'].clip(upper=df['cvr'].quantile(0.95))
df['cpc'] = df['cpc'].clip(upper=df['cpc'].quantile(0.95))
df['cpa'] = df['cpa'].clip(upper=df['cpa'].quantile(0.95))
df['bid_amount'] = df['bid_amount'].clip(lower=10, upper=500)

# Handle infinite values from division
df[['ctr', 'cvr', 'cpc', 'cpa']] = df[['ctr', 'cvr', 'cpc', 'cpa']].replace([np.inf, -np.inf], 0)
print("Data Cleaning: Handled outliers and infinite values.")

# Step 4: Exploratory Data Analysis (EDA)
# Analyze distributions.
print("EDA: Summary statistics...")
print(df.describe())

# Visualize conversions by season
fig_eda = px.box(
    df,
    x='season',
    y='conversions',
    title='Conversions by Season',
    labels={'season': 'Season', 'conversions': 'Conversions'}
)
fig_eda.update_layout(template='plotly_white')
fig_eda.write_html('eda_conversions_by_season.html')
print("EDA: Generated box plot of conversions by season.")

# Step 5: Feature Engineering
# Encode categorical features and select features for modeling.
le_destination = LabelEncoder()
le_age_group = LabelEncoder()
le_income_level = LabelEncoder()
le_season = LabelEncoder()

df['destination_encoded'] = le_destination.fit_transform(df['destination'])
df['audience_age_group_encoded'] = le_age_group.fit_transform(df['audience_age_group'])
df['audience_income_level_encoded'] = le_income_level.fit_transform(df['audience_income_level'])
df['season_encoded'] = le_season.fit_transform(df['season'])

# Features for classification
features = [
    'destination_encoded', 'hotel_rating', 'bid_amount', 'impressions', 'clicks', 'conversions',
    'ctr', 'cvr', 'cpc', 'cpa', 'ad_relevance_score', 'audience_age_group_encoded',
    'audience_income_level_encoded', 'season_encoded'
]

print("Feature Engineering: Encoded categorical features and computed performance metrics.")

# Step 6: Model Selection
# Choose Random Forest Classifier for performance prediction.
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Model Selection: Chose Random Forest Classifier.")

# Step 7: Model Training
# Split data and train.
X = df[features]
y = df['performance_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Model Training: Trained Random Forest Classifier.")

# Step 8: Model Evaluation
# Evaluate with accuracy, precision, recall, F1-score.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Evaluation: Performance metrics...")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Step 9: Model Tuning
# Grid search for hyperparameter optimization.
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Model Tuning: Best parameters:", grid_search.best_params_)

# Re-evaluate with best model
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print("Model Evaluation (Tuned): Performance metrics...")
print(f"Accuracy: {accuracy_best:.2f}")
print(f"Precision: {precision_best:.2f}")
print(f"Recall: {recall_best:.2f}")
print(f"F1-Score: {f1_best:.2f}")

# Step 10: Bid Optimization
# Predict performance and adjust bids.
df['predicted_performance'] = best_model.predict(X)
df['probability_high_performance'] = best_model.predict_proba(X)[:, 1]

# Bid adjustment formula
def adjust_bid(row, budget_constraint=0.9):
    base_bid = row['bid_amount']
    ctr_norm = min(row['ctr'] / df['ctr'].max(), 1)  # Normalize CTR
    cvr_norm = min(row['cvr'] / df['cvr'].max(), 1)  # Normalize CVR
    relevance_norm = row['ad_relevance_score']  # Already 0–1
    season_factor = 1.2 if row['season'] == 'Peak' else 0.8
    # Weights for features
    w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
    # Adjustment factor
    adjustment = w1 * ctr_norm + w2 * cvr_norm + w3 * relevance_norm + w4 * season_factor
    # New bid with performance probability
    new_bid = base_bid * adjustment * (1 + row['probability_high_performance']) * budget_constraint
    # Cap bids
    return round(min(max(new_bid, 10), 500), 2)

df['optimized_bid'] = df.apply(adjust_bid, axis=1)

# Step 11: Prediction and Insights
# Identify top 15 high-performing campaigns and top 5 underperforming campaigns.
top_15_campaigns = df[df['predicted_performance'] == 1][['campaign_id', 'ad_name', 'destination', 'month', 'hotel_rating', 'ctr', 'cvr', 'cpa', 'optimized_bid', 'probability_high_performance']].sort_values(by='probability_high_performance', ascending=False).head(15)
top_5_underperforming = df[df['predicted_performance'] == 0][['campaign_id', 'ad_name', 'destination', 'month', 'hotel_rating', 'ctr', 'cvr', 'cpa', 'optimized_bid', 'probability_high_performance']].sort_values(by='probability_high_performance').head(5)

# Save outputs
df.to_csv('ad_campaign_predictions.csv', index=False)
top_15_campaigns.to_csv('top_15_campaigns.csv', index=False)
top_5_underperforming.to_csv('top_5_underperforming_campaigns.csv', index=False)

print("\nTop 15 High-Performing Campaigns:")
print(top_15_campaigns[['campaign_id', 'ad_name', 'destination', 'month', 'hotel_rating', 'ctr', 'cvr', 'cpa', 'optimized_bid']])

print("\nTop 5 Underperforming Campaigns (Budget Burners):")
print(top_5_underperforming[['campaign_id', 'ad_name', 'destination', 'month', 'hotel_rating', 'ctr', 'cvr', 'cpa', 'optimized_bid']])

# Step 12: Visualization
# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred_best)
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=['Low-Performing', 'High-Performing'],
    y=['Low-Performing', 'High-Performing'],
    colorscale='Blues',
    showscale=True
)
fig_cm.update_layout(title='Confusion Matrix', template='plotly_white')
fig_cm.write_html('confusion_matrix.html')

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

fig_importance = px.bar(
    feature_importance,
    x='importance',
    y='feature',
    title='Feature Importance in Random Forest Classifier',
    labels={'importance': 'Importance', 'feature': 'Feature'},
    orientation='h'
)
fig_importance.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})
fig_importance.write_html('feature_importance.html')
print("Visualization: Generated confusion matrix and feature importance plots.")

# Step 13: Deployment
# Simulate Flask API for real-time bid prediction.
app = Flask(__name__)

@app.route('/predict_bid', methods=['POST'])
def predict_bid():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df['destination_encoded'] = le_destination.transform(input_df['destination'])
    input_df['audience_age_group_encoded'] = le_age_group.transform(input_df['audience_age_group'])
    input_df['audience_income_level_encoded'] = le_income_level.transform(input_df['audience_income_level'])
    input_df['season_encoded'] = le_season.transform(input_df['season'])
    input_features = input_df[features]
    performance = best_model.predict(input_features)[0]
    prob_high = best_model.predict_proba(input_features)[0, 1]
    optimized_bid = adjust_bid(input_df.iloc[0])
    return jsonify({
        'predicted_performance': int(performance),
        'probability_high_performance': round(prob_high, 2),
        'optimized_bid': optimized_bid
    })

# Simulate running Flask app (commented out)
# if __name__ == '__main__':
#     app.run(debug=True)
print("Deployment: Simulated Flask API for real-time bid prediction.")

# Step 14: Monitoring and Maintenance
# Plan to monitor and update model.
print("Monitoring and Maintenance: Monitor campaign performance, collect new ad data, and retrain periodically.")