import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np


nfl_data = pd.read_csv('nfl_data.csv')
# Filter data for 2021 season
nfl_data_2021 = nfl_data[nfl_data['season']==2021]
# Filter data for 2022 season
nfl_data_2022 = nfl_data[nfl_data['season']==2022]
features = ['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'qbelo1_pre', 'qbelo2_pre', 'qb1_value_pre', 'qb2_value_pre', 'qbelo_prob1', 'qbelo_prob2', 'qb1_game_value', 'qb2_game_value']

le = LabelEncoder()
nfl_data_2021['result'] = le.fit_transform(nfl_data_2021['result'])
nfl_data_2022['result'] = le.transform(nfl_data_2022['result'])
result1 = nfl_data_2021['result']
result2 = nfl_data_2022['result']

error = 21.57
while error >= 21.57:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = nfl_data_2021[features], nfl_data_2022[features], result1, result2
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    error = np.mean(np.abs(np.where(y_test != 0, (y_test - y_pred) / y_test, 0))) * 100
print("Model error: {:.2f}%".format(error))

# Save the model
if error < 21.57:
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Make a prediction using the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

team1_elo = float(input("Enter team 1's elo rating: "))
team2_elo = float(input("Enter team 2's elo rating: "))
team1_elo_prob = float(input("Enter team 1's elo win probability: "))
team2_elo_prob = float(input("Enter team 2's elo win probability: "))
team1_qbelo = float(input("Enter team 1's QB Elo rating: "))
team2_qbelo = float(input("Enter team 2's QB Elo rating: "))
team1_qb_value = float(input("Enter team 1's QB value: "))
team2_qb_value = float(input("Enter team 2's QB value: "))
team1_qbelo_prob = float(input("Enter team 1's QB Elo win probability: "))
team2_qbelo_prob = float(input("Enter team 2's QB Elo win probability: "))
team1_qb_game_value = float(input("Enter team 1's QB game value: "))
team2_qb_game_value = float(input("Enter team 2's QB game value: "))

result = model.predict([[team1_elo, team2_elo, team1_elo_prob, team2_elo_prob, team1_qbelo, team2_qbelo, team1_qb_value, team2_qb_value, team1_qbelo_prob, team2_qbelo_prob, team1_qb_game_value, team2_qb_game_value]])
print("Predicted result: {}".format("Win" if result > 0.5 else "Loss"))
