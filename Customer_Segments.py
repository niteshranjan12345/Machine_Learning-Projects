import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import uuid
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate Synthetic Dataset
n_customers = 100000
locations = ['New York', 'Paris', 'Tokyo', 'London', 'Dubai', 'Sydney', 'Miami', 'Rome', 'Singapore', 'Barcelona']
affinity_audiences = ['Luxury Travel', 'Adventure Travel', 'Business Travel', 'Family Travel', 'Budget Travel']
life_event_audiences = ['Wedding', 'Job Change', 'Moving', 'Retirement', 'None']
in_market_audiences = ['Hotel Booking', 'Travel Packages', 'Car Rental', 'Flight Booking', 'None']
genders = ['Male', 'Female', 'Other']
occupations = ['Professional', 'Manager', 'Entrepreneur', 'Student', 'Retired']
income_levels = ['Low', 'Medium', 'High']
loyalty_statuses = ['None', 'Silver', 'Gold', 'Platinum']

# Create customer data
data = {
    'customer_id': [str(uuid.uuid4()) for _ in range(n_customers)],
    'age': np.random.randint(18, 80, n_customers),
    'gender': np.random.choice(genders, n_customers, p=[0.48, 0.48, 0.04]),
    'income_level': np.random.choice(income_levels, n_customers, p=[0.3, 0.5, 0.2]),
    'location': np.random.choice(locations, n_customers),
    'occupation': np.random.choice(occupations, n_customers, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
    'affinity_audience': np.random.choice(affinity_audiences, n_customers),
    'life_event_audience': np.random.choice(life_event_audiences, n_customers, p=[0.1, 0.1, 0.1, 0.05, 0.65]),
    'in_market_audience': np.random.choice(in_market_audiences, n_customers, p=[0.3, 0.2, 0.1, 0.1, 0.3]),
    'booking_frequency': np.random.randint(0, 20, n_customers),
    'avg_spend': np.random.uniform(50, 2000, n_customers).round(2),
    'loyalty_status': np.random.choice(loyalty_statuses, n_customers, p=[0.5, 0.3, 0.15, 0.05])
}

df = pd.DataFrame(data)

# Generate profitability label (High if high income, frequent bookings, or luxury affinity)
df['profitability'] = 'Low'
high_profit_conditions = (
    (df['income_level'] == 'High') |
    (df['booking_frequency'] > 10) |
    (df['avg_spend'] > 1000) |
    (df['affinity_audience'] == 'Luxury Travel') |
    (df['loyalty_status'].isin(['Gold', 'Platinum']))
)
df.loc[high_profit_conditions, 'profitability'] = 'High'

# Save synthetic dataset to CSV
df.to_csv('customer_segments_synthetic.csv', index=False)

# 2. Preprocessing for Random Forest Classifier
# Encode categorical features
label_encoders = {}
categorical_cols = ['gender', 'income_level', 'location', 'occupation', 
                   'affinity_audience', 'life_event_audience', 'in_market_audience', 'loyalty_status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
features = ['age', 'gender', 'income_level', 'location', 'occupation', 
            'affinity_audience', 'life_event_audience', 'in_market_audience', 
            'booking_frequency', 'avg_spend', 'loyalty_status']
X = df[features]
y = df['profitability'].apply(lambda x: 1 if x == 'High' else 0)  # Encode High=1, Low=0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")

# Predict profitability for all customers
df['profitability_pred'] = rf_model.predict(X)
df['profitability_prob'] = rf_model.predict_proba(X)[:, 1]  # Probability of High profitability

# Save predictions to CSV
df.to_csv('customer_profitability_predictions.csv', index=False)

# 4. Identify Top 50 Customers
# Select top 50 customers by predicted probability of High profitability
top_50_customers = df.sort_values(by='profitability_prob', ascending=False).head(50)

# Decode categorical features for interpretability
for col in categorical_cols:
    top_50_customers[col] = label_encoders[col].inverse_transform(top_50_customers[col])

# Save top 50 customers to CSV
top_50_customers.to_csv('top_50_profitable_customers.csv', index=False)

# Print top 50 customers
print("\nTop 50 Customers with Highest Predicted Profitability:")
print(top_50_customers[['customer_id', 'affinity_audience', 'life_event_audience', 
                        'in_market_audience', 'income_level', 'profitability_prob']])

# 5. Visualizations
# Bar plot: Distribution of audience types among top 50 customers
audience_types = ['affinity_audience', 'life_event_audience', 'in_market_audience']
fig_audience = go.Figure()
for audience in audience_types:
    counts = top_50_customers[audience].value_counts()
    fig_audience.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            name=audience.replace('_audience', '').title()
        )
    )

fig_audience.update_layout(
    title='Audience Type Distribution Among Top 50 Profitable Customers',
    xaxis_title='Audience Type',
    yaxis_title='Count',
    barmode='group',
    template='plotly_white'
)
fig_audience.write_html('audience_distribution_top_50.html')

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

fig_importance = px.bar(
    feature_importance,
    x='importance',
    y='feature',
    title='Feature Importance in Random Forest Classifier',
    labels={'importance': 'Importance', 'feature': 'Feature'},
    orientation='h'
)
fig_importance.update_layout(
    template='plotly_white',
    yaxis={'categoryorder': 'total ascending'}
)
fig_importance.write_html('feature_importance.html')