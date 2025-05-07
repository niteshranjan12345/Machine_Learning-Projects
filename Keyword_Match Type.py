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
n_keywords_per_type = 100
india_destinations = [
    'Goa', 'Mumbai', 'Delhi', 'Jaipur', 'Bangalore', 'Kolkata', 'Chennai', 
    'Hyderabad', 'Kerala', 'Udaipur', 'Shimla', 'Manali', 'Darjeeling', 
    'Agra', 'Rishikesh'
]
general_keywords = [
    'online hotel booking', 'cheap hotels near me', 'luxury hotels', 
    'budget hotels', 'hotel deals', 'best hotels', 'family hotels', 
    'business hotels', 'hotel reservations', 'last minute hotels', 
    '5 star hotels', 'pet friendly hotels', 'beach hotels', 
    'hotel rooms', 'affordable hotels'
]
keyword_modifiers = [
    '', 'near me', 'luxury', 'budget', 'cheap', 'best', 'deals', 
    '5 star', 'family', 'business', 'last minute', 'pet friendly', 
    'beach', 'affordable'
]

# Generate keywords for each match type
keywords = []
for match_type in ['Broad', 'Exact', 'Phrase']:
    for _ in range(n_keywords_per_type):
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

# Add features
keywords_df['search_volume_per_month'] = np.random.randint(100, 50000, len(keywords_df))
keywords_df['competition'] = np.random.choice(['Low', 'Medium', 'High'], len(keywords_df), p=[0.4, 0.4, 0.2])
keywords_df['competition_index'] = np.random.randint(0, 100, len(keywords_df))
keywords_df['top_of_page_low'] = np.random.uniform(0.1, 5.0, len(keywords_df)).round(2)
keywords_df['top_of_page_high'] = keywords_df['top_of_page_low'] + np.random.uniform(0.5, 5.0, len(keywords_df)).round(2)
keywords_df['keyword_length'] = keywords_df['keyword'].apply(lambda x: len(x.strip('[]\"').split()))

# Generate synthetic high_lead_potential label
keywords_df['high_lead_potential'] = 0
high_lead_conditions = (
    (keywords_df['search_volume_per_month'] > 5000) & 
    (keywords_df['competition_index'] < 70) & 
    (keywords_df['top_of_page_high'] < 3.5)
)
keywords_df.loc[high_lead_conditions, 'high_lead_potential'] = 1

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
y = keywords_df['high_lead_potential']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")

# Predict lead potential probability for all keywords
keywords_df['lead_potential_prob'] = rf_model.predict_proba(X)[:, 1]

# Decode categorical features for output
keywords_df['competition'] = le_competition.inverse_transform(keywords_df['competition'])
keywords_df['match_type'] = le_match_type.inverse_transform(keywords_df['match_type'])

# Save predictions to CSV
keywords_df.to_csv('keyword_predictions_rf.csv', index=False)

# 4. Identify Top 50 Keywords
# Select top 50 keywords by lead potential probability
top_50_keywords = keywords_df.sort_values(by='lead_potential_prob', ascending=False).head(50)

# Save top 50 keywords to CSV
top_50_keywords.to_csv('top_50_keywords_rf.csv', index=False)

# Print top 50 keywords
print("\nTop 50 Keywords with Highest Predicted Lead Potential:")
print(top_50_keywords[['keyword', 'match_type', 'search_volume_per_month', 
                       'competition', 'competition_index', 'top_of_page_high', 
                       'lead_potential_prob']])

# 5. Visualizations
# Bar plot: Distribution of match types among top 50 keywords
match_type_counts = top_50_keywords['match_type'].value_counts()
fig_match_type = px.bar(
    x=match_type_counts.index,
    y=match_type_counts.values,
    title='Match Type Distribution Among Top 50 High-Lead-Potential Keywords',
    labels={'x': 'Match Type', 'y': 'Count'},
)
fig_match_type.update_layout(
    template='plotly_white',
    xaxis_title='Match Type',
    yaxis_title='Count'
)
fig_match_type.write_html('match_type_distribution_top_50.html')

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