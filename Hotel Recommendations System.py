import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Dataset
np.random.seed(42)
n_users = 500
n_hotels = 15
n_reviews = 1000

user_ids = np.random.randint(0, n_users, n_reviews * n_hotels)
hotel_ids = np.repeat(np.arange(n_hotels), n_reviews)
ratings = np.random.normal(3.5, 1.0, n_reviews * n_hotels).clip(1, 5).round()
# Simulate sentiment (1 = positive, 0 = neutral/negative)
sentiments = np.random.binomial(1, 0.7, n_reviews * n_hotels)  # 70% positive reviews
locations = np.random.choice(['Mumbai', 'Goa', 'Delhi', 'Jaipur', 'Bengaluru', 'Kerala', 'Udaipur', 
                             'Shimla', 'Chennai', 'Rishikesh', 'Hyderabad', 'Ooty', 'Agra', 'Pune', 'Manali'], 
                             n_reviews * n_hotels)
prices = np.random.choice(['Low', 'Medium', 'High'], n_reviews * n_hotels, p=[0.3, 0.4, 0.3])

data = pd.DataFrame({
    'user_id': user_ids,
    'hotel_id': hotel_ids,
    'rating': ratings,
    'sentiment': sentiments,
    'location': locations,
    'price': prices
})

# Encode categorical variables
user_encoder = LabelEncoder()
hotel_encoder = LabelEncoder()
location_encoder = LabelEncoder()
price_encoder = LabelEncoder()

data['user_id'] = user_encoder.fit_transform(data['user_id'])
data['hotel_id'] = hotel_encoder.fit_transform(data['hotel_id'])
data['location'] = location_encoder.fit_transform(data['location'])
data['price'] = price_encoder.fit_transform(data['price'])

# Step 2: Prepare Data for Training
n_users = len(user_encoder.classes_)
n_hotels = len(hotel_encoder.classes_)
n_locations = len(location_encoder.classes_)
n_prices = len(price_encoder.classes_)

X_user = data['user_id'].values
X_hotel = data['hotel_id'].values
X_location = data['location'].values
X_price = data['price'].values
y = data['rating'].values

# Train-test split
from sklearn.model_selection import train_test_split
X_user_train, X_user_test, X_hotel_train, X_hotel_test, X_location_train, X_location_test, \
X_price_train, X_price_test, y_train, y_test = train_test_split(
    X_user, X_hotel, X_location, X_price, y, test_size=0.2, random_state=42
)

# Step 3: Build Deep Learning Model (Neural Collaborative Filtering)
embedding_size = 50

# User input and embedding
user_input = Input(shape=(1,))
user_embedding = Embedding(n_users, embedding_size)(user_input)
user_flat = Flatten()(user_embedding)

# Hotel input and embedding
hotel_input = Input(shape=(1,))
hotel_embedding = Embedding(n_hotels, embedding_size)(hotel_input)
hotel_flat = Flatten()(hotel_embedding)

# Location input and embedding
location_input = Input(shape=(1,))
location_embedding = Embedding(n_locations, embedding_size)(location_input)
location_flat = Flatten()(location_embedding)

# Price input and embedding
price_input = Input(shape=(1,))
price_embedding = Embedding(n_prices, embedding_size)(price_input)
price_flat = Flatten()(price_embedding)

# Concatenate all embeddings
concat = Concatenate()([user_flat, hotel_flat, location_flat, price_flat])
dense = Dense(128, activation='relu')(concat)
dense = Dropout(0.2)(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dropout(0.2)(dense)
output = Dense(1)(dense)

model = Model(inputs=[user_input, hotel_input, location_input, price_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Step 4: Train the Model
history = model.fit(
    [X_user_train, X_hotel_train, X_location_train, X_price_train], y_train,
    validation_data=([X_user_test, X_hotel_test, X_location_test, X_price_test], y_test),
    epochs=20, batch_size=32, verbose=1
)

# Step 5: Evaluate and Predict
y_pred = model.predict([X_user_test, X_hotel_test, X_location_test, X_price_test])
mse = np.mean((y_pred.flatten() - y_test) ** 2)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Step 6: Visualize Training History
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Example Recommendation
def recommend_hotel(user_id, top_n=5):
    user_id_encoded = user_encoder.transform([user_id])[0]
    all_hotels = np.array(range(n_hotels))
    all_locations = np.random.choice(range(n_locations), n_hotels)
    all_prices = np.random.choice(range(n_prices), n_hotels)
    predictions = model.predict([np.full(n_hotels, user_id_encoded), all_hotels, all_locations, all_prices])
    top_indices = np.argsort(predictions.flatten())[-top_n:][::-1]
    return [(hotel_encoder.inverse_transform([i])[0], predictions[i][0]) for i in top_indices]

# Test recommendation for a sample user
sample_user_id = 0
recommendations = recommend_hotel(sample_user_id)
print("\nTop 5 Hotel Recommendations for User", sample_user_id, ":")
for hotel_id, score in recommendations:
    print(f"Hotel ID: {hotel_id}, Predicted Rating: {score:.2f}")