from flask import Flask, request, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
# https://flask-cors.readthedocs.io/en/latest/api.html
CORS(app, resources=["/api/*"])

@app.route('/')
def index():
    return '<p>Hello, world</p><p>finally working</p>'


@app.get('/match-result')
def get_match_result():
    home = request.args["home"]
    away = request.args["away"]

    overall_win_rates = pd.read_csv("static/overall_win_rates.csv")
    h2h_win_rates = pd.read_csv("static/h2h_win_rates.csv")
    # overall_win_rates = pd.read_csv(url_for('static', filename='overall_win_rates.csv'))
    # h2h_win_rates = pd.read_csv(url_for('static', filename='h2h_win_rates.csv'))

    home_home_wr = 0
    home_away_wr = 0
    away_home_wr = 0
    away_away_wr = 0

    for _, row in overall_win_rates.iterrows():
        if (row['team'] == home):
            home_home_wr = row['home']
            home_away_wr = row['away']
        if (row['team'] == away):
            away_home_wr = row['home']
            away_away_wr = row['away']

    home_vs_away_wr = 0
    away_vs_home_wr = 0

    for _, row in h2h_win_rates.iterrows():
        if (row['home_team'] == home and row['away_team'] == away):
            home_vs_away_wr = row['home_wr']
            away_vs_home_wr = row['away_wr']
    
    match_features = np.array([[home_home_wr, home_away_wr, away_home_wr, away_away_wr, home_vs_away_wr, away_vs_home_wr]])

    model = joblib.load("static/Logistic Regression.pkl")
    # model = joblib.load(url_for('static', filename='Logistic Regression.pkl'))
    prob = model.predict_proba(match_features)[0]
    pred = model.predict(match_features)[0]

    outcomes = {0: f"home win", 1: f"away win", 2: "drow"}

    return {
        "home_winning_probability": prob[0],
        "away_winning_probability": prob[1],
        "draw_probability": prob[2],
        "expected_outcome": outcomes[pred],
    }


if __name__ == "__main__":
    app.run(debug=True)