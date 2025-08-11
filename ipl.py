import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle

# Step 1: Load dataset
try:
    data = pd.read_csv(r"D:\IPL_Dataset(2008-2024).csv")
    print("Dataset loaded successfully.")
    print("\nDataset columns:", list(data.columns))
    print("\nFirst 5 rows of dataset:\n", data.head())
except FileNotFoundError:
    print("Error: File not found at 'D:\\IPL_Dataset(2008-2024).csv'. Exiting.")
    exit(1)

# Step 2: Preprocess and feature engineering
team_name_mapping = {
    'Chennai Super Kings': 'CSK',
    'Mumbai Indians': 'MI',
    'Royal Challengers Bengaluru': 'RCB',
    'Royal Challengers Bangalore': 'RCB',
    'Kolkata Knight Riders': 'KKR',
    'Sunrisers Hyderabad': 'SRH',
    'Delhi Capitals': 'DC',
    'Rajasthan Royals': 'RR',
    'Punjab Kings': 'PBKS',
    'Lucknow Super Giants': 'LSG',
    'Gujarat Titans': 'GT'
}

# Split Teams column
data[['team1', 'team2']] = data['Teams'].str.split(' vs ', expand=True)
data['team1'] = data['team1'].str.strip().map(team_name_mapping)
data['team2'] = data['team2'].str.strip().map(team_name_mapping)
data['Toss_Winner'] = data['Toss_Winner'].map(team_name_mapping)
data['Match_Winner'] = data['Match_Winner'].map(team_name_mapping)

# Drop rows with missing mappings
data = data.dropna(subset=['team1', 'team2', 'Toss_Winner', 'Match_Winner'])
print(f"\nDataset shape after dropping missing team mappings: {data.shape}")

# Filter for 2023-2024 matches
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data[data['Date'].dt.year.isin([2023, 2024])]
print(f"Filtered dataset shape (2023-2024): {data.shape}")

# Home venue mapping
home_venues = {
    'CSK': 'MA Chidambaram Stadium, Chepauk, Chennai',
    'MI': 'Wankhede Stadium, Mumbai',
    'RCB': 'M Chinnaswamy Stadium, Bengaluru',
    'KKR': 'Eden Gardens, Kolkata',
    'SRH': 'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
    'DC': 'Arun Jaitley Stadium, Delhi',
    'RR': 'Sawai Mansingh Stadium, Jaipur',
    'PBKS': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'LSG': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'GT': 'Narendra Modi Stadium, Ahmedabad'
}

# Features
data['toss_win'] = (data['Toss_Winner'] == data['team1']).astype(int)
data['win'] = (data['Match_Winner'] == data['team1']).astype(int)

data['home_advantage'] = data.apply(
    lambda row: 1 if home_venues.get(row['team1'], '') in row['Venue'] else 0, axis=1
)

team1_batting = data.groupby('team1')['First_Innings_Score'].mean().to_dict()
team2_batting = data.groupby('team2')['First_Innings_Score'].mean().to_dict()
data['batting_strength_team1'] = data['team1'].map(team1_batting)
data['batting_strength_team2'] = data['team2'].map(team2_batting)
data['batting_strength_diff'] = data['batting_strength_team1'] - data['batting_strength_team2']

team1_bowling = data.groupby('team2')['First_Innings_Score'].mean().to_dict()
team2_bowling = data.groupby('team1')['First_Innings_Score'].mean().to_dict()
data['bowling_strength_team1'] = data['team1'].map(team1_bowling)
data['bowling_strength_team2'] = data['team2'].map(team2_bowling)
data['bowling_strength_diff'] = data['bowling_strength_team2'] - data['bowling_strength_team1']

def calculate_h2h_win_rate(row, data):
    team1, team2 = row['team1'], row['team2']
    past_matches = data[((data['team1'] == team1) & (data['team2'] == team2)) |
                       ((data['team1'] == team2) & (data['team2'] == team1))]
    if len(past_matches) == 0:
        return 0.5
    team1_wins = len(past_matches[past_matches['Match_Winner'] == team1])
    return team1_wins / len(past_matches)

data['head_to_head_win_rate'] = data.apply(lambda row: calculate_h2h_win_rate(row, data), axis=1)

# Step 3: Prepare data
features = ['batting_strength_diff', 'bowling_strength_diff', 'home_advantage', 'head_to_head_win_rate']
match_features = features + ['toss_win']

X_match = data[match_features]
y_match = data['win']
X_toss = data[features]
y_toss = data['toss_win']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_match, y_match, test_size=0.2, random_state=42)
X_toss_train, X_toss_test, y_toss_train, y_toss_test = train_test_split(X_toss, y_toss, test_size=0.2, random_state=42)

# Step 4: Train XGBoost models
match_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
match_model.fit(X_train, y_train)

toss_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
toss_model.fit(X_toss_train, y_toss_train)

# Step 5: Evaluate
match_pred = match_model.predict(X_test)
match_acc = accuracy_score(y_test, match_pred)
print(f"Match Model Accuracy: {match_acc:.4f}")

toss_pred = toss_model.predict(X_toss_test)
toss_acc = accuracy_score(y_toss_test, toss_pred)
print(f"Toss Model Accuracy: {toss_acc:.4f}")

# Step 6: Prediction for user input
valid_teams = sorted(set(team_name_mapping.values()))
print("\nAvailable teams:", valid_teams)
team1 = input("Enter Team 1 (e.g., CSK): ").strip().upper()
team2 = input("Enter Team 2 (e.g., MI): ").strip().upper()
venue = input("Enter venue: ").strip()

if team1 not in valid_teams or team2 not in valid_teams:
    print(f"Error: Choose from {valid_teams}")
    exit(1)

new_data = pd.DataFrame({
    'batting_strength_diff': [team1_batting.get(team1, 150) - team2_batting.get(team2, 150)],
    'bowling_strength_diff': [team2_bowling.get(team2, 150) - team1_bowling.get(team1, 150)],
    'home_advantage': [1 if home_venues.get(team1, '') == venue else 0],
    'head_to_head_win_rate': [data[((data['team1'] == team1) & (data['team2'] == team2)) |
                                  ((data['team1'] == team2) & (data['team2'] == team1))]
                             ['Match_Winner'].eq(team1).mean() or 0.5]
})

# Predict toss
toss_prob = toss_model.predict_proba(new_data)[0][1]
toss_winner = team1 if toss_prob > 0.5 else team2
print(f"\nToss win probability for {team1}: {toss_prob:.4f}")
print(f"Predicted Toss Winner: {toss_winner}")

# Predict match
new_data['toss_win'] = [1 if toss_winner == team1 else 0]
match_prob = match_model.predict_proba(new_data[match_features])[0][1]
winning_team = team1 if match_prob > 0.5 else team2
print(f"\nMatch win probability for {team1}: {match_prob:.4f}")
print(f"Predicted Match Winner: {winning_team}")

# Step 7: Save models
with open('ipl_match_xgb.pkl', 'wb') as f:
    pickle.dump(match_model, f)
with open('ipl_toss_xgb.pkl', 'wb') as f:
    pickle.dump(toss_model, f)

print("\nModels saved as 'ipl_match_xgb.pkl' and 'ipl_toss_xgb.pkl'")
