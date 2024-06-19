import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def create_testdataset(team1_players, team2_players, player_id_csv):
    player_id_dict = dict(zip(player_id_csv['player name'], player_id_csv['player_ID']))
    
    match_data = pd.DataFrame(columns=['Match ID'] + [f'Team {i+1} P{j+1}' for i in range(2) for j in range(8)] +
                                        ['Team 1 Goals', 'Team 2 Goals', 'Team 1 Result', 'Team 2 Result'])
    # put player names into columns
    for team_name, players in zip(['Team 1', 'Team 2'], [team1_players, team2_players]):
        for i, player in enumerate(players):
            match_data.at[0, f'{team_name} P{i+1}'] = player

    match_data.fillna(0, inplace=True)
    # put actual player IDs into columns
    for column in match_data.columns:
        if column.startswith('Team 1 P') or column.startswith('Team 2 P'):
            match_data[column] = match_data[column].apply(lambda x: player_id_dict.get(x, 0))
    
    return match_data

def training_model(processed_training_dataset):
    df = processed_training_dataset.select_dtypes(exclude=['object'])
    x_train = df.drop(["Match ID", "Team 1 Result", "Team 2 Result", "Team 1 Goals", "Team 2 Goals"], axis=1)
    y_train = df[["Team 1 Goals", "Team 2 Goals", "Team 1 Result"]]

    model = make_pipeline(StandardScaler(), MultiOutputRegressor(RandomForestRegressor()))
    model.fit(x_train, y_train)
    return model

def predict_outcomes(processed_test_dataset, model):
    df = processed_test_dataset.select_dtypes(exclude=['object'])
    x_test = df.drop(["Match ID", "Team 1 Result", "Team 2 Result", "Team 1 Goals", "Team 2 Goals"], axis=1)
    predictions = model.predict(x_test)
    
    team1_goals = predictions[:, 0]
    team2_goals = predictions[:, 1]
    team1_win_prob = predictions[:, 2]
    team2_win_prob = 1 - team1_win_prob

    team1_win = (team1_win_prob > 0.5).astype(int)
    
    return team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob

def train_test_models(processed_training_dataset, processed_test_dataset):
    model = training_model(processed_training_dataset)
    team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob = predict_outcomes(processed_test_dataset, model)
    return team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob

def print_results(team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob):
    print("Predicted Goals:")
    print("Team 1:", team1_goals)
    print("Team 2:", team2_goals)
    print("Team 1 Win Probability:", team1_win_prob)
    print("Team 2 Win Probability:", team2_win_prob)
    #print("Draw predicted:", draw)
    print("\nPredicted Results:")
    print("Team 1 Wins" if team1_win == 1 else "Team 2 Wins" if team1_win == 0 else "Draw")