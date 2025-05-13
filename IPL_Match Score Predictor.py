import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Simulated IPL dataset (200 matches)
np.random.seed(42)
n_samples = 200

data = {
    'batting_team_strength': np.random.uniform(50, 100, n_samples),  # Team batting strength (0-100)
    'bowling_team_strength': np.random.uniform(50, 100, n_samples),  # Team bowling strength (0-100)
    'venue_impact': np.random.uniform(0.8, 1.2, n_samples),        # Venue factor (0.8-1.2)
    'overs_played': np.random.uniform(15, 20, n_samples),          # Overs completed (15-20)
    'actual_score': np.random.uniform(120, 220, n_samples)         # Actual first innings score
}

# Simulate competitor teams and outcomes
competitors = ['CSK', 'MI', 'RCB', 'KKR', 'DC']
df = pd.DataFrame(data)
df['competitor'] = np.random.choice(competitors, n_samples)
df['target_score'] = df['actual_score'] * np.random.uniform(0.9, 1.1, n_samples)  # Target for second innings
df['win_loss'] = (df['actual_score'] > df['target_score']).astype(int)  # 1 = win, 0 = loss

# Features and target
X = df[['batting_team_strength', 'bowling_team_strength', 'venue_impact', 'overs_played']].values
y = df['actual_score'].values

# Split data using DataFrame indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 1: Build Deep Learning Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Hidden layer 1
    Dense(32, activation='relu'),  # Hidden layer 2
    Dense(16, activation='relu'),  # Hidden layer 3
    Dense(1)  # Output layer (regression for score)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_data=(X_test_scaled, y_test), verbose=1)

# Step 2: Evaluate and predict
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Mean Absolute Error: {mae:.2f}")

y_pred = model.predict(X_test_scaled, verbose=0).flatten()

# Determine win/loss for predictions
test_df = pd.DataFrame(X_test, columns=['batting_team_strength', 'bowling_team_strength', 'venue_impact', 'overs_played'])
test_df['actual_score'] = y_test
test_df['predicted_score'] = y_pred
test_df['competitor'] = df['competitor'].iloc[idx_test]  # Use idx_test for correct indexing
test_df['target_score'] = df['target_score'].iloc[idx_test]
test_df['win_loss_pred'] = (test_df['predicted_score'] > test_df['target_score']).astype(int)

# Step 3: Graphics
# Bar chart for actual vs predicted scores
plt.figure(figsize=(10, 6))
sample_indices = np.random.choice(len(y_test), 10, replace=False)  # Sample 10 predictions
plt.bar(np.arange(10) - 0.2, y_test[sample_indices], 0.4, label='Actual Score', color='#FF5722')
plt.bar(np.arange(10) + 0.2, y_pred[sample_indices], 0.4, label='Predicted Score', color='#2196F3')
plt.xlabel('Sample Matches')
plt.ylabel('Score')
plt.title('Actual vs Predicted Scores for Sample Matches')
plt.xticks(np.arange(10), [f'Match {i+1}' for i in range(10)])
plt.legend()
for i, (act, pred) in enumerate(zip(y_test[sample_indices], y_pred[sample_indices])):
    plt.text(i - 0.2, act + 2, f'{act:.0f}', ha='center')
    plt.text(i + 0.2, pred + 2, f'{pred:.0f}', ha='center')
plt.show()

# Bar chart for win/loss against competitors
win_loss_by_comp = test_df.groupby('competitor')['win_loss_pred'].value_counts().unstack(fill_value=0)
win_loss_by_comp.columns = ['Loss', 'Win']
win_loss_by_comp.plot(kind='bar', figsize=(10, 6), color=['#FF9800', '#4CAF50'])
plt.title('Win/Loss Prediction Against Competitors')
plt.xlabel('Competitor')
plt.ylabel('Count')
plt.xticks(rotation=0)
for i, (win, loss) in enumerate(zip(win_loss_by_comp['Win'], win_loss_by_comp['Loss'])):
    plt.text(i - 0.1, win + 0.5, str(win), ha='center')
    plt.text(i + 0.1, loss + 0.5, str(loss), ha='center')
plt.legend(title='Outcome')
plt.show()

# Step 4: Examples of predictions
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_data = X_test[sample_indices]
sample_actual = y_test[sample_indices]
sample_pred = y_pred[sample_indices]
sample_comp = test_df['competitor'].iloc[sample_indices].values
sample_target = test_df['target_score'].iloc[sample_indices].values

print("\nSample Predictions:")
for i in range(len(sample_data)):
    print(f"Sample {i+1}:")
    print(f"Features: Batting={sample_data[i][0]:.2f}, Bowling={sample_data[i][1]:.2f}, Venue={sample_data[i][2]:.2f}, Overs={sample_data[i][3]:.2f}")
    print(f"Actual Score: {sample_actual[i]:.0f}, Predicted Score: {sample_pred[i]:.0f}, Target: {sample_target[i]:.0f}")
    print(f"Competitor: {sample_comp[i]}, Outcome: {'Win' if sample_pred[i] > sample_target[i] else 'Loss'}\n")