import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic dataset
n_interactions = 50000
hotel_keywords = [
    'hotel booking', 'cheap hotels', 'luxury hotels', 'hotel deals', 
    'book hotel online', 'hotel reservations', 'best hotel prices', 
    'hotel rooms', 'vacation hotels', 'hotel discounts'
]

data = {
    'Keyword': np.random.choice(hotel_keywords, n_interactions),
    'Click_Timestamp': np.random.uniform(0, 86400, n_interactions),  # Seconds in a day
    'IP_Frequency': np.random.poisson(2, n_interactions),  # Most IPs click 1-3 times
    'User_Agent_Score': np.random.uniform(0.3, 1, n_interactions),  # Legit users higher
    'Click_Duration': np.random.exponential(60, n_interactions),  # Seconds, legit users longer
    'Conversion': np.random.choice([0, 1], n_interactions, p=[0.9, 0.1])  # 10% conversion rate
}

df = pd.DataFrame(data)

# Simulate fraud (10% of data)
fraud_idx = np.random.choice(n_interactions, int(0.1 * n_interactions), replace=False)
df.loc[fraud_idx, 'IP_Frequency'] = np.random.poisson(10, len(fraud_idx))  # High IP frequency
df.loc[fraud_idx, 'User_Agent_Score'] = np.random.uniform(0, 0.3, len(fraud_idx))  # Bot-like
df.loc[fraud_idx, 'Click_Duration'] = np.random.uniform(0, 10, len(fraud_idx))  # Short duration
df['Is_Fraud'] = 0
df.loc[fraud_idx, 'Is_Fraud'] = 1

# Save dataset (optional)
df.to_csv('hotel_ad_campaign_dataset.csv', index=False)
print("Dataset Preview:")
print(df.head())
print(f"Fraud rate: {df['Is_Fraud'].mean():.2%}")

# Step 2: Preprocessing
features = ['Click_Timestamp', 'IP_Frequency', 'User_Agent_Score', 'Click_Duration']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['Iso_Fraud_Score'] = iso_forest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
df['Iso_Fraud_Score'] = df['Iso_Fraud_Score'].replace({1: 0, -1: 1})  # Convert to 0 = normal, 1 = fraud

# Step 4: Autoencoder
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, verbose=0)

# Reconstruction error
X_pred = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.square(X_scaled - X_pred), axis=1)
df['AE_Fraud_Score'] = (mse > np.percentile(mse, 90)).astype(int)  # Top 10% errors = fraud

# Step 5: Combine fraud predictions (ensemble: fraud if either model flags it)
df['Predicted_Fraud'] = (df['Iso_Fraud_Score'] | df['AE_Fraud_Score']).astype(int)

# Step 6: Evaluate (using ground truth)
print("\nFraud Detection Performance:")
print(f"Isolation Forest Accuracy: {np.mean(df['Is_Fraud'] == df['Iso_Fraud_Score']):.4f}")
print(f"Autoencoder Accuracy: {np.mean(df['Is_Fraud'] == df['AE_Fraud_Score']):.4f}")
print(f"Combined Accuracy: {np.mean(df['Is_Fraud'] == df['Predicted_Fraud']):.4f}")

# Step 7: Top keywords with fraudulent clicks/conversions
fraud_by_keyword = df[df['Predicted_Fraud'] == 1].groupby('Keyword').agg({
    'Predicted_Fraud': 'sum',  # Count of fraudulent clicks
    'Conversion': 'sum'  # Fraudulent conversions
}).rename(columns={'Predicted_Fraud': 'Fraudulent_Clicks'})
fraud_by_keyword['Fraud_Rate'] = fraud_by_keyword['Fraudulent_Clicks'] / df.groupby('Keyword').size()
top_fraud_keywords = fraud_by_keyword.sort_values(by='Fraudulent_Clicks', ascending=False)

print("\nTop Keywords with Fraudulent Clicks and Conversions:")
print(top_fraud_keywords)

# Save results
top_fraud_keywords.to_csv('top_fraud_keywords.csv')

# Step 8: Visualization
plt.figure(figsize=(10, 6))
top_fraud_keywords['Fraudulent_Clicks'].plot(kind='bar')
plt.title('Top Keywords by Fraudulent Clicks')
plt.xlabel('Keyword')
plt.ylabel('Number of Fraudulent Clicks')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fraudulent_clicks_by_keyword.png')
plt.show()