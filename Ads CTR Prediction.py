import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic dataset
n_impressions = 100000
data = {
    'Time_Spent_On_Site': np.random.uniform(10, 300, n_impressions),  # Seconds
    'Pages_Visited': np.random.randint(1, 20, n_impressions),
    'Previous_Clicks': np.random.randint(0, 50, n_impressions),
    'User_Age': np.random.randint(18, 65, n_impressions),
    'Ad_Category': np.random.choice(['Tech', 'Fashion', 'Food', 'Travel'], n_impressions),
    'Ad_Length': np.random.uniform(5, 60, n_impressions),  # Seconds
    'Ad_Keyword_Relevance': np.random.uniform(0, 1, n_impressions),
    'Ad_Visibility_Score': np.random.uniform(0, 1, n_impressions)
}

df = pd.DataFrame(data)

# Simulate CTR based on heuristic rules with noise
df['CTR'] = (
    0.3 * df['Time_Spent_On_Site'] / 300 + 
    0.2 * df['Previous_Clicks'] / 50 + 
    0.2 * df['Ad_Keyword_Relevance'] + 
    0.1 * df['Ad_Visibility_Score'] + 
    np.random.normal(0, 0.05, n_impressions)  # Noise
).clip(0, 1)  # Ensure CTR is between 0 and 1

# One-hot encode categorical variable
df = pd.get_dummies(df, columns=['Ad_Category'], drop_first=True)

# Save dataset (optional)
df.to_csv('ad_ctr_dataset.csv', index=False)
print("Dataset Preview:")
print(df.head())
print(f"Average CTR: {df['CTR'].mean():.4f}")

# Step 2: Data Preprocessing
X = df.drop('CTR', axis=1)
y = df['CTR']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_xgb):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_xgb):.4f}")

# Step 4: Train LightGBM Model
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lgb = lgb_model.predict(X_test)
print("\nLightGBM Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lgb):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lgb):.4f}")

# Step 5: Visualizations
# Feature Importance (XGBoost)
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig('xgb_feature_importance.png')
plt.show()

# Feature Importance (LightGBM)
plt.figure(figsize=(10, 6))
lgb.plot_importance(lgb_model, max_num_features=10)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig('lgb_feature_importance.png')
plt.show()

# Predicted vs Actual CTR (LightGBM)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lgb, alpha=0.5, s=10)
plt.plot([0, 1], [0, 1], 'r--', lw=2)  # Diagonal line
plt.xlabel("Actual CTR")
plt.ylabel("Predicted CTR")
plt.title("LightGBM: Predicted vs Actual CTR")
plt.tight_layout()
plt.savefig('ctr_pred_vs_actual.png')
plt.show()

# Save predictions (optional)
results = pd.DataFrame({
    'Actual_CTR': y_test,
    'Predicted_CTR_XGB': y_pred_xgb,
    'Predicted_CTR_LGB': y_pred_lgb
})
results.to_csv('ctr_predictions.csv', index=False)