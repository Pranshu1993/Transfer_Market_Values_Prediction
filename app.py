from flask import Flask, request, jsonify, render_template
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np


# load the saved model
model = tf.keras.models.load_model('models/feed_forward_NN.h5')

# initialize Flask app
app = Flask(__name__)
#run
# define a function to preprocess the input data
def preprocess_input_data(minutes_played_overall, minutes_played_home, minutes_played_away, appearances_overall, appearances_home, appearances_away, goals_overall, goals_home, goals_away, assists_overall, assists_home, assists_away, clean_sheets_overall, clean_sheets_away, conceded_overall, conceded_home, conceded_away, rank_in_league_top_midfielders, skill_moves, crossing, finishing, short_passing, volleys, dribbling, sprint_speed, stamina, penalties, sliding_tackle, gk_kicking, gk_reflexes, rank_in_club_top_scorer,gk_handling,aggression,standing_tackle,gk_positioning, penalty_goals,min_per_match,defense_awareness_marking,year):
    # create a pandas dataframe with the input data
    # input_dict = {"A":[minutes_played_overall, minutes_played_home, minutes_played_away, appearances_overall, appearances_home, appearances_away, goals_overall, goals_home, goals_away, assists_overall, assists_home, assists_away, clean_sheets_overall, clean_sheets_away, conceded_overall, conceded_home, conceded_away, rank_in_league_top_midfielders, skill_moves, crossing, finishing, short_passing, volleys, dribbling, sprint_speed, stamina, penalties, sliding_tackle, gk_kicking, gk_reflexes, aggression, rank_in_club_top_scorer, year, gk_positioning, penalty_goals, gk_handling, defense_awareness_marking, min_per_match, standing_tackle]
    input_dict = pd.DataFrame({
    'minutes_played_overall': [minutes_played_overall],
    'minutes_played_home': [minutes_played_home],
    'minutes_played_away': [minutes_played_away],
    'appearances_overall': [appearances_overall],
    'appearances_home': [appearances_home],
    'appearances_away': [appearances_away],
    'goals_overall': [goals_overall],
    'goals_home': [goals_home],
    'goals_away': [goals_away],
    'assists_overall': [assists_overall],
    'assists_home': [assists_home],
    'assists_away': [assists_away],
    'clean_sheets_overall': [clean_sheets_overall],
    'clean_sheets_away': [clean_sheets_away],
    'conceded_overall': [conceded_overall],
    'conceded_home': [conceded_home],
    'conceded_away': [conceded_away],
    'rank_in_league_top_midfielders': [rank_in_league_top_midfielders],
    'Skill Moves': [skill_moves],
    'Crossing': [crossing],
    'Finishing': [finishing],
    'Short Passing': [short_passing],
    'Volleys': [volleys],
    'Dribbling': [dribbling],
    'Sprint Speed': [sprint_speed],
    'Stamina': [stamina],
    'Penalties': [penalties],
    'Sliding Tackle': [sliding_tackle],
    'GK Kicking': [gk_kicking],
    'GK Reflexes': [gk_reflexes],
    'rank_in_club_top_scorer': [rank_in_club_top_scorer],
    'Standing Tackle': [standing_tackle],
    'Aggression': [aggression],
    'min_per_match': [min_per_match],
    'penalty_goals': [penalty_goals],
    'GK Positioning': [gk_positioning],
    'GK Handling': [gk_handling],
    'Defense Awareness/Marking': [defense_awareness_marking],
    

        
    })
<<<<<<< Updated upstream
    input_data= pd.DataFrame.from_dict(input_dict)
=======
    #input_data = pd.DataFrame(input_dict, index=[0])
    input_data=np.array(input_dict)
    print(type(input_data))
    # input_data= pd.DataFrame.from_dict(input_dict)
>>>>>>> Stashed changes
    # input_data = pd.DataFrame(list(input_dict.values())).T
# scale the input data using the same scaler used during training
    # years=year
    # print(years)
    # scaler = load('scaler.joblib')
    # input_data_scaled = scaler.transform(input_data)
    # print(input_data_scaled)
    # print(type(input_data_scaled))
    # input_data_scaled = input_data_scaled.append(years)
    # print(input_data_scaled)
    
    years = np.array([year])  # Convert year to a numpy array
    print(years)
    scaler = load('scaler.joblib')
    input_data_scaled = scaler.transform(input_data)
    print(input_data_scaled)
    print(type(input_data_scaled))
    input_data_scaled = np.hstack((input_data_scaled, years.reshape(-1, 1)))
    print(input_data_scaled)

    return input_data_scaled


@app.route('/')
def home():

    return render_template('index.html')

# define a Flask route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # get the input data from the request
# Example format for processing form input and storing in variables

    name = request.form['Name']
    minutes_played_overall = float(request.form['minutes_played_overall'])
    minutes_played_home = float(request.form['minutes_played_home'])
    minutes_played_away = float(request.form['minutes_played_away'])
    appearances_overall = float(request.form['appearances_overall'])
    appearances_home = float(request.form['appearances_home'])
    appearances_away = float(request.form['appearances_away'])
    goals_overall = float(request.form['goals_overall'])
    goals_home = float(request.form['goals_home'])
    goals_away = float(request.form['goals_away'])
    assists_overall = float(request.form['assists_overall'])
    assists_home = float(request.form['assists_home'])
    assists_away = float(request.form['assists_away'])
    clean_sheets_overall = float(request.form['clean_sheets_overall'])
    clean_sheets_away = float(request.form['clean_sheets_away'])
    conceded_overall = float(request.form['conceded_overall'])
    conceded_home = float(request.form['conceded_home'])
    conceded_away = float(request.form['conceded_away'])
    rank_in_league_top_midfielders = float(request.form['rank_in_league_top_midfielders'])
    skill_moves = float(request.form['skill_moves'])
    crossing = float(request.form['crossing'])
    finishing = float(request.form['finishing'])
    short_passing = float(request.form['short_passing'])
    volleys = float(request.form['volleys'])
    dribbling = float(request.form['dribbling'])
    sprint_speed = float(request.form['sprint_speed'])
    stamina = float(request.form['stamina'])
    penalties = float(request.form['penalties'])
    sliding_tackle = float(request.form['sliding_tackle'])
    gk_kicking = float(request.form['gk_kicking'])
    gk_reflexes = float(request.form['gk_reflexes'])
    aggression = float(request.form['aggression'])
    rank_in_club_top_scorer = float(request.form['rank_in_club_top_scorer'])
    year = float(request.form['year'])
    gk_positioning = float(request.form['gk_positioning'])
    penalty_goals = float(request.form['penalty_goals'])
    gk_handling = float(request.form['gk_handling'])
    defense_awareness_marking = float(request.form['defense_awareness_marking'])
    min_per_match = float(request.form['min_per_match'])
    standing_tackle = float(request.form['standing_tackle'])

    # preprocess the input data
    input_data_scaled = preprocess_input_data(minutes_played_overall, minutes_played_home, minutes_played_away, appearances_overall, appearances_home, appearances_away, goals_overall, goals_home, goals_away, assists_overall, assists_home, assists_away, clean_sheets_overall, clean_sheets_away, conceded_overall, conceded_home, conceded_away, rank_in_league_top_midfielders, skill_moves, crossing, finishing, short_passing, volleys, dribbling, sprint_speed, stamina, penalties, sliding_tackle, gk_kicking, gk_reflexes, rank_in_club_top_scorer,gk_handling,aggression,standing_tackle,gk_positioning, penalty_goals,min_per_match,defense_awareness_marking,year)
    # make a prediction
    prediction = model.predict(input_data_scaled)[0][0]
    # format the output as a JSON object
    output = { 
        'Name': name,
        'Prediction': prediction
    }

    return render_template('index.html', prediction_text='Price Prediction for {} is  €{} '.format(name, prediction))

# start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

