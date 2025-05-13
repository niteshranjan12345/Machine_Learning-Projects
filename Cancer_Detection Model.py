import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Simulate TCIA-like dataset (200 samples)
np.random.seed(42)
n_samples = 200

# Simulated parameters inspired by TCIA imaging features
data = {
    'tumor_size': np.random.uniform(1, 10, n_samples),         # Tumor size in cm
    'texture_heterogeneity': np.random.uniform(0.1, 1.0, n_samples),  # Texture variation
    'intensity_variance': np.random.uniform(50, 200, n_samples),     # Image intensity variation
    'patient_age': np.random.uniform(30, 80, n_samples)       # Patient age
}

df = pd.DataFrame(data)

# Simulate cancer danger label (1 = dangerous, 0 = not dangerous) based on threshold
df['is_dangerous'] = ((df['tumor_size'] > 5) | 
                      (df['texture_heterogeneity'] > 0.7) | 
                      (df['intensity_variance'] > 150) | 
                      (df['patient_age'] > 60)).astype(int)

# Features and target
X = df[['tumor_size', 'texture_heterogeneity', 'intensity_variance', 'patient_age']].values
y = df['is_dangerous'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 1: Build Deep Learning Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Hidden layer 1
    Dense(32, activation='relu'),  # Hidden layer 2
    Dense(16, activation='relu'),  # Hidden layer 3
    Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_data=(X_test_scaled, y_test), verbose=1)

# Step 2: Evaluate and predict
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.2f}")

y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Create test DataFrame with predictions
test_df = pd.DataFrame(X_test, columns=['tumor_size', 'texture_heterogeneity', 'intensity_variance', 'patient_age'])
test_df['actual_danger'] = y_test
test_df['predicted_danger'] = y_pred

# Step 3: Identify and print parameters causing cancer
def identify_cancer_parameters(row):
    params = []
    if row['tumor_size'] > 5:
        params.append('tumor_size > 5 cm')
    if row['texture_heterogeneity'] > 0.7:
        params.append('high texture_heterogeneity > 0.7')
    if row['intensity_variance'] > 150:
        params.append('high intensity_variance > 150')
    if row['patient_age'] > 60:
        params.append('patient_age > 60')
    return ', '.join(params) if params else 'No significant parameters'

test_df['cancer_parameters'] = test_df.apply(identify_cancer_parameters, axis=1)
print("\nParameters Causing Cancer Detection:")
for index, row in test_df.iterrows():
    print(f"Sample {index+1}: {row['cancer_parameters']} -> Predicted: {'Dangerous' if row['predicted_danger'] == 1 else 'Not Dangerous'}, Actual: {'Dangerous' if row['actual_danger'] == 1 else 'Not Dangerous'}")

# Step 4: Graphics
# Bar chart for parameter impact (average values for dangerous vs not dangerous)
danger_params = test_df[test_df['actual_danger'] == 1].mean()
safe_params = test_df[test_df['actual_danger'] == 0].mean()
param_names = ['tumor_size', 'texture_heterogeneity', 'intensity_variance', 'patient_age']

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(param_names))
plt.bar(index, danger_params[param_names], bar_width, label='Dangerous', color='#FF5722')
plt.bar(index + bar_width, safe_params[param_names], bar_width, label='Not Dangerous', color='#2196F3')
plt.xlabel('Parameters')
plt.ylabel('Average Value')
plt.title('Parameter Impact on Cancer Detection (Dangerous vs Not Dangerous)')
plt.xticks(index + bar_width / 2, param_names, rotation=45)
plt.legend()
for i, (d, s) in enumerate(zip(danger_params[param_names], safe_params[param_names])):
    plt.text(i, d + 0.1, f'{d:.1f}', ha='center')
    plt.text(i + bar_width, s + 0.1, f'{s:.1f}', ha='center')
plt.show()

# Pie chart for dangerous vs not dangerous classification
danger_count = test_df['predicted_danger'].sum()
safe_count = len(test_df) - danger_count
plt.figure(figsize=(6, 6))
plt.pie([danger_count, safe_count], labels=['Dangerous', 'Not Dangerous'], colors=['#FF5722', '#2196F3'], autopct='%1.1f%%')
plt.title('Predicted Cancer Detection Outcome')
plt.show()

# Step 5: Example predictions
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_data = X_test[sample_indices]
sample_actual = y_test[sample_indices]
sample_pred = y_pred[sample_indices]
sample_params = test_df['cancer_parameters'].iloc[sample_indices].values

print("\nSample Predictions:")
for i in range(len(sample_data)):
    print(f"Sample {i+1}:")
    print(f"Features: Tumor Size={sample_data[i][0]:.2f}, Texture={sample_data[i][1]:.2f}, Intensity={sample_data[i][2]:.2f}, Age={sample_data[i][3]:.2f}")
    print(f"Actual: {'Dangerous' if sample_actual[i] == 1 else 'Not Dangerous'}, Predicted: {'Dangerous' if sample_pred[i] == 1 else 'Not Dangerous'}")
    print(f"Parameters: {sample_params[i]}\n")