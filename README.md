# IPL-Toss-Match-Winner-Prediction


This project predicts IPL toss winners and match winners using match data from 2008 to 2024.
It uses XGBoost for improved accuracy compared to Logistic Regression, leveraging historical statistics and team performance metrics.

📂 Dataset
Matches from 2008 to 2024

Includes team names, venue, toss winner, match winner, scores, and dates.

🛠 Features Used
Batting strength difference (avg. first innings score)

Bowling strength difference

Home advantage (based on venue)

Head-to-head win rate

Toss win indicator (for match prediction)

🔍 How It Works
Preprocessing:

Maps full team names to abbreviations.

Handles name changes (e.g., RCB/Bangalore).

Calculates statistical features from historical data.

Model Training:

Two separate XGBoost models:

Toss prediction model.

Match prediction model (uses toss prediction as input).

Trains using recent data (2023–2024) for better relevance.

Prediction:

User inputs: Team 1, Team 2, and venue.

Model outputs:

Probability of each team winning the toss.

Probability of each team winning the match.

Saving Models:

Models saved as ipl_match_xgb.pkl and ipl_toss_xgb.pkl.

🚀 How to Run
bash
Copy
Edit
pip install pandas scikit-learn xgboost
python ipl.py
📊 Example Output
yaml
Copy
Edit
Toss win probability for CSK: 0.6421
Predicted Toss Winner: CSK

Match win probability for CSK: 0.7012
Predicted Match Winner: CSK
