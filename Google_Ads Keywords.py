import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import uuid
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate Synthetic Dataset
n_keywords = 100000
india_destinations = [
    'Goa', 'Mumbai', 'Delhi', 'Jaipur', 'Bangalore', 'Kolkata', 'Chennai', 
    'Hyderabad', 'Kerala', 'Udaipur', 'Shimla', 'Manali', 'Darjeeling', 
    'Agra', 'Rishikesh', 'Pune', 'Ahmedabad', 'Varanasi', 'Amritsar', 'Coorg'
]
general_keywords = [
    'online hotel booking', 'cheap hotels near me', 'luxury hotels', 
    'budget hotels', 'hotel deals', 'best hotels', 'family hotels', 
    'business hotels', 'hotel reservations', 'last minute hotels', 
    '5 star hotels', 'pet friendly hotels', 'beach hotels', 
    'hotel rooms', 'affordable hotels', 'hotel discounts', 
    'hotel packages', 'romantic hotels', 'spa hotels', 'resort hotels'
]
keyword_modifiers = [
    '', 'near me', 'luxury', 'budget', 'cheap', 'best', 'deals', 
    '5 star', 'family', 'business', 'last minute', 'pet friendly', 
    'beach', 'affordable', 'discounts', 'packages', 'romantic', 'spa', 'resort'
]

# Generate keywords
keywords = []
for _ in range(n_keywords):
    if np.random.random() < 0.6:  # 60% location-based
        city = np.random.choice(india_destinations)
        modifier = np.random.choice(keyword_modifiers)
        keyword = f"hotels in {city} {modifier}".strip()
        is_location_based = 1
    else:  # 40% general
        base_keyword = np.random.choice(general_keywords)
        modifier = np.random.choice(keyword_modifiers)
        keyword = f"{base_keyword} {modifier}".strip() if modifier else base_keyword
        is_location_based = 0
    keywords.append({'keyword': keyword, 'is_location_based': is_location_based})

keywords_df = pd.DataFrame(keywords)

# Add features
keywords_df['search_volume_per_month'] = np.random.randint(100, 100000, n_keywords)
keywords_df['competition'] = np.random.choice(['Low', 'Medium', 'High'], n_keywords, p=[0.4, 0.4, 0.2])
keywords_df['competition_index'] = np.random.randint(0, 100, n_keywords)
keywords_df['top_of_page_low'] = np.random.uniform(0.1, 5.0, n_keywords).round(2)
keywords_df['top_of_page_high'] = keywords_df['top_of_page_low'] + np.random.uniform(0.5, 5.0, n_keywords).round(2)
keywords_df['keyword_length'] = keywords_df['keyword'].apply(lambda x: len(x.split()))

# Generate synthetic high_conversion label
keywords_df['high_conversion'] = 0
high_conversion_conditions = (
    (keywords_df['search_volume_per_month'] > 5000) & 
    (keywords_df['competition_index'] < 60) & 
    (keywords_df['top_of_page_high'] < 3.0)
)
keywords_df.loc[high_conversion_conditions, 'high_conversion'] = 1

# Save synthetic dataset to CSV
keywords_df.to_csv('hotel_booking_keywords_synthetic.csv', index=False)

# 2. Preprocessing for Random Forest Classifier
# Encode categorical features
le = LabelEncoder()
keywords_df['competition'] = le.fit_transform(keywords_df['competition'])

# Features and target
features = ['search_volume_per_month', 'competition', 'competition_index', 
            'top_of_page_low', 'top_of_page_high', 'keyword_length', 'is_location_based']
X = keywords_df[features]
y = keywords_df['high_conversion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")

# Predict conversion probability for all keywords
keywords_df['conversion_prob'] = rf_model.predict_proba(X)[:, 1]

# 4. Define Logic and Formula for Best Keywords
# Scoring formula: High volume, low competition, low bids
max_volume = keywords_df['search_volume_per_month'].max()
keywords_df['keyword_score'] = (
    (keywords_df['search_volume_per_month'] / max_volume) * 
    (1 - keywords_df['competition_index'] / 100) * 
    (1 / (keywords_df['top_of_page_high'] + 1))
)

# Save predictions to CSV
keywords_df.to_csv('keyword_predictions_rf.csv', index=False)

# 5. Identify Top 100 Keywords
# Select top 100 keywords by conversion probability
top_100_keywords = keywords_df.sort_values(by='conversion_prob', ascending=False).head(100)

# Save top 100 keywords to CSV
top_100_keywords.to_csv('top_100_keywords_rf.csv', index=False)

# Print top 100 keywords
print("\nTop 100 Keywords with Highest Predicted Conversion Probability:")
print(top_100_keywords[['keyword', 'search_volume_per_month', 'competition', 
                        'competition_index', 'top_of_page_high', 'conversion_prob', 'keyword_score']])

# 6. Visualizations
# Scatter plot: Search volume vs. competition index, colored by conversion probability
fig_scatter = px.scatter(
    top_100_keywords,
    x='search_volume_per_month',
    y='competition_index',
    color='conversion_prob',
    size='keyword_score',
    hover_data=['keyword', 'top_of_page_high'],
    title='Top 100 Keywords: Search Volume vs. Competition Index',
    labels={
        'search_volume_per_month': 'Search Volume per Month',
        'competition_index': 'Competition Index',
        'conversion_prob': 'Conversion Probability'
    }
)
fig_scatter.update_layout(
    template='plotly_white',
    coloraxis_colorbar_title='Conversion Prob'
)
fig_scatter.write_html('keyword_performance_scatter.html')

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