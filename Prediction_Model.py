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
n_keywords = 5000
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
match_types = ['Broad', 'Exact', 'Phrase']

# Generate keywords
keywords = []
for _ in range(n_keywords):
    match_type = np.random.choice(match_types)
    if np.random.random() < 0.7:  # 70% location-based
        city = np.random.choice(india_destinations)
        modifier = np.random.choice(keyword_modifiers)
        base_keyword = f"hotels in {city} {modifier}".strip()
        is_location_based = 1
    else:  # 30% general
        base_keyword = np.random.choice(general_keywords)
        modifier = np.random.choice(keyword_modifiers)
        base_keyword = f"{base_keyword} {modifier}".strip() if modifier else base_keyword
        is_location_based = 0
    
    # Format keyword by match type
    if match_type == 'Exact':
        keyword = f"[{base_keyword}]"
    elif match_type == 'Phrase':
        keyword = f"\"{base_keyword}\""
    else:  # Broad
        keyword = base_keyword
    
    keywords.append({
        'keyword': keyword,
        'match_type': match_type,
        'is_location_based': is_location_based
    })

keywords_df = pd.DataFrame(keywords)

# Add features (bid prices in INR, 1 USD ≈ 83 INR)
keywords_df['search_volume_per_month'] = np.random.randint(100, 100000, n_keywords)
keywords_df['competition'] = np.random.choice(['Low', 'Medium', 'High'], n_keywords, p=[0.4, 0.4, 0.2])
keywords_df['competition_index'] = np.random.randint(0, 100, n_keywords)
keywords_df['top_of_page_low'] = np.random.uniform(8, 400, n_keywords).round(2)  # ₹8–₹400
keywords_df['top_of_page_high'] = keywords_df['top_of_page_low'] + np.random.uniform(40, 400, n_keywords).round(2)  # ₹40–₹800
keywords_df['keyword_length'] = keywords_df['keyword'].apply(lambda x: len(x.strip('[]\"').split()))

# Generate synthetic high_click_conversion label
keywords_df['high_click_conversion'] = 0
high_click_conversion_conditions = (
    (keywords_df['search_volume_per_month'] > 10000) & 
    (keywords_df['competition_index'] < 60) & 
    (keywords_df['top_of_page_high'] < 250)  # ₹250 ≈ $3
)
keywords_df.loc[high_click_conversion_conditions, 'high_click_conversion'] = 1

# Save synthetic dataset to CSV
keywords_df.to_csv('hotel_booking_keywords_synthetic.csv', index=False)

# 2. Preprocessing for Random Forest Classifier
# Encode categorical features
le_competition = LabelEncoder()
keywords_df['competition'] = le_competition.fit_transform(keywords_df['competition'])
le_match_type = LabelEncoder()
keywords_df['match_type'] = le_match_type.fit_transform(keywords_df['match_type'])

# Features and target
features = ['search_volume_per_month', 'competition', 'competition_index', 
            'top_of_page_low', 'top_of_page_high', 'keyword_length', 
            'is_location_based', 'match_type']
X = keywords_df[features]
y = keywords_df['high_click_conversion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")

# Predict click and conversion probability for all keywords
keywords_df['click_conversion_prob'] = rf_model.predict_proba(X)[:, 1]

# 4. Estimate Clicks and Conversions
# Heuristic: Clicks = search_volume * click-through rate (CTR, adjusted by competition and match type)
# Conversions = clicks * conversion rate (weighted by click_conversion_prob)
ctr_by_competition = {'Low': 0.05, 'Medium': 0.03, 'High': 0.01}
ctr_by_match_type = {'Broad': 1.2, 'Exact': 0.8, 'Phrase': 1.0}  # Multipliers

keywords_df['decoded_competition'] = le_competition.inverse_transform(keywords_df['competition'])
keywords_df['decoded_match_type'] = le_match_type.inverse_transform(keywords_df['match_type'])

keywords_df['estimated_clicks'] = keywords_df.apply(
    lambda row: int(
        row['search_volume_per_month'] * 
        ctr_by_competition[row['decoded_competition']] * 
        ctr_by_match_type[row['decoded_match_type']]
    ), 
    axis=1
)
keywords_df['estimated_conversions'] = keywords_df.apply(
    lambda row: int(row['estimated_clicks'] * (row['click_conversion_prob'] * 0.05)),  # Base conversion rate 5%
    axis=1
)

# Drop temporary columns
keywords_df = keywords_df.drop(['decoded_competition', 'decoded_match_type'], axis=1)

# Save predictions to CSV
keywords_df.to_csv('keyword_predictions_rf.csv', index=False)

# 5. Identify Top 100 Keywords
# Select top 100 keywords by click_conversion_prob
top_100_keywords = keywords_df.sort_values(by='click_conversion_prob', ascending=False).head(100)

# Decode categorical features for output
top_100_keywords['competition'] = le_competition.inverse_transform(top_100_keywords['competition'])
top_100_keywords['match_type'] = le_match_type.inverse_transform(top_100_keywords['match_type'])

# Save top 100 keywords to CSV
top_100_keywords.to_csv('top_100_keywords_rf.csv', index=False)

# Print top 100 keywords
print("\nTop 100 Keywords with Highest Predicted Click and Conversion Potential:")
print(top_100_keywords[['keyword', 'match_type', 'search_volume_per_month', 
                        'competition', 'competition_index', 'top_of_page_high', 
                        'click_conversion_prob', 'estimated_clicks', 'estimated_conversions']])

# 6. Visualizations
# Scatter plot: Search volume vs. competition index, colored by click_conversion_prob
fig_scatter = px.scatter(
    top_100_keywords,
    x='search_volume_per_month',
    y='competition_index',
    color='click_conversion_prob',
    size='estimated_clicks',
    hover_data=['keyword', 'match_type', 'estimated_conversions'],
    title='Top 100 Keywords: Search Volume vs. Competition Index',
    labels={
        'search_volume_per_month': 'Search Volume per Month',
        'competition_index': 'Competition Index',
        'click_conversion_prob': 'Click & Conversion Probability'
    }
)
fig_scatter.update_layout(
    template='plotly_white',
    coloraxis_colorbar_title='Probability'
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