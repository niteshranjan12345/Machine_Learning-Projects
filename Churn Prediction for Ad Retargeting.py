import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, request, jsonify
import uuid
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Market Research
# Understand factors driving churn: low engagement, long time since last booking, low loyalty.
print("Market Research: Identified key features - website visits, ad clicks, past bookings, recency, loyalty.")

# Step 2: Data Collection
# Generate synthetic dataset for 12,000 hotel customers.
n_customers = 12000
locations = ['Goa', 'Mumbai', 'Delhi', 'Jaipur', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Kerala', 'Udaipur']
genders = ['Male', 'Female', 'Other']

data = {
    'customer_id': [str(uuid.uuid4()) for _ in range(n_customers)],
    'age': np.random.randint(18, 80, n_customers),
    'gender': np.random.choice(genders, n_customers, p=[0.48, 0.48, 0.04]),
    'location': np.random.choice(locations, n_customers),
    'website_visits': np.random.randint(0, 50, n_customers),
    'ad_clicks': np.random.randint(0, 20, n_customers),
    'past_bookings': np.random.randint(0, 10, n_customers),
    'avg_booking_value': np.random.uniform(2000, 50000, n_customers).round(2),  # INR
    'last_booking_days_ago': np.random.randint(1, 730, n_customers),  # Up to 2 years
    'session_duration': np.random.uniform(0, 300, n_customers).round(2)  # Minutes
}

df = pd.DataFrame(data)

# Generate loyalty score
df['loyalty_score'] = (df['past_bookings'] * 10 + df['website_visits'] * 0.5 + df['ad_clicks'] * 1).clip(upper=100)

# Generate churn status (1 = churn, 0 = no churn)
# Churn if low engagement (few visits/clicks), long time since last booking, low loyalty
df['churn_status'] = ((df['website_visits'] < 5) & (df['ad_clicks'] < 3) & (df['last_booking_days_ago'] > 180) & (df['loyalty_score'] < 30)).astype(int)

# Save synthetic dataset
df.to_csv('hotel_customer_churn.csv', index=False)
print("Data Collection: Generated synthetic dataset for 12,000 hotel customers.")

# Step 3: Data Cleaning
# Handle missing values and outliers.
print("Data Cleaning: Checking for missing values...")
print(df.isnull().sum())

# Cap outliers
df['avg_booking_value'] = df['avg_booking_value'].clip(upper=df['avg_booking_value'].quantile(0.95))
df['last_booking_days_ago'] = df['last_booking_days_ago'].clip(upper=730)
df['session_duration'] = df['session_duration'].clip(upper=df['session_duration'].quantile(0.95))
df[['website_visits', 'ad_clicks', 'past_bookings']] = df[['website_visits', 'ad_clicks', 'past_bookings']].clip(lower=0)

print("Data Cleaning: Handled outliers and ensured positive values.")

# Step 4: Exploratory Data Analysis (EDA)
# Analyze distributions and churn patterns.
print("EDA: Summary statistics...")
print(df.describe())

# Visualize churn by last_booking_days_ago
fig_eda = px.box(
    df,
    x='churn_status',
    y='last_booking_days_ago',
    title='Last Booking Days Ago by Churn Status',
    labels={'churn_status': 'Churn Status (1=Churn, 0=No Churn)', 'last_booking_days_ago': 'Days Since Last Booking'}
)
fig_eda.update_layout(template='plotly_white')
fig_eda.write_html('eda_churn_by_recency.html')
print("EDA: Generated box plot of recency by churn status.")

# Step 5: Feature Engineering
# Encode categorical features and scale numerical features.
le_gender = LabelEncoder()
le_location = LabelEncoder()

df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['location_encoded'] = le_location.fit_transform(df['location'])

# Features for classification
features = [
    'age', 'gender_encoded', 'location_encoded', 'website_visits', 'ad_clicks',
    'past_bookings', 'avg_booking_value', 'last_booking_days_ago',
    'session_duration', 'loyalty_score'
]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

print("Feature Engineering: Encoded categorical features and scaled numerical features.")

# Step 6: Model Selection
# Choose XGBoost Classifier for churn prediction.
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
print("Model Selection: Chose XGBoost Classifier.")

# Step 7: Model Training
# Split data and train.
X = X_scaled
y = df['churn_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Model Training: Trained XGBoost Classifier.")

# Step 8: Model Evaluation
# Evaluate with accuracy, precision, recall, F1-score, ROC-AUC.
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Model Evaluation: Performance metrics...")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Step 9: Model Tuning
# Grid search for hyperparameter optimization.
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Model Tuning: Best parameters:", grid_search.best_params_)

# Re-evaluate with best model
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_proba_best)

print("Model Evaluation (Tuned): Performance metrics...")
print(f"Accuracy: {accuracy_best:.2f}")
print(f"Precision: {precision_best:.2f}")
print(f"Recall: {recall_best:.2f}")
print(f"F1-Score: {f1_best:.2f}")
print(f"ROC-AUC: {roc_auc_best:.2f}")

# Step 10: Prediction and Insights
# Predict churn probabilities and identify top 15 high-risk customers.
df['churn_probability'] = best_model.predict_proba(X_scaled)[:, 1] * 100  # Convert to percentage
top_15_high_churn = df[['customer_id', 'age', 'gender', 'location', 'website_visits', 'ad_clicks', 'past_bookings', 'last_booking_days_ago', 'churn_probability']].sort_values(by='churn_probability', ascending=False).head(15)

# Save outputs
df.to_csv('customer_churn_predictions.csv', index=False)
top_15_high_churn.to_csv('top_15_high_churn_customers.csv', index=False)

print("\nTop 15 Customers with Highest Churn Probability:")
print(top_15_high_churn[['customer_id', 'age', 'gender', 'location', 'website_visits', 'ad_clicks', 'past_bookings', 'last_booking_days_ago', 'churn_probability']])

# Step 11: Visualization
# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc_best:.2f})'))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(
    title='ROC Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    template='plotly_white'
)
fig_roc.write_html('roc_curve.html')

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

fig_importance = px.bar(
    feature_importance,
    x='importance',
    y='feature',
    title='Feature Importance in XGBoost Classifier',
    labels={'importance': 'Importance', 'feature': 'Feature'},
    orientation='h'
)
fig_importance.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})
fig_importance.write_html('feature_importance.html')
print("Visualization: Generated ROC curve and feature importance plots.")

# Step 12: Deployment
# Simulate Flask API for real-time churn prediction.
app = Flask(__name__)

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df['gender_encoded'] = le_gender.transform(input_df['gender'])
    input_df['location_encoded'] = le_location.transform(input_df['location'])
    input_features = input_df[features]
    input_scaled = scaler.transform(input_features)
    churn_prob = best_model.predict_proba(input_scaled)[0, 1] * 100
    return jsonify({'churn_probability': round(churn_prob, 2)})

# Simulate running Flask app (commented out)
# if __name__ == '__main__':
#     app.run(debug=True)
print("Deployment: Simulated Flask API for real-time churn prediction.")

# Step 13: Monitoring and Maintenance
# Plan to monitor and update model.
print("Monitoring and Maintenance: Monitor churn predictions, collect new customer data, and retrain periodically.")