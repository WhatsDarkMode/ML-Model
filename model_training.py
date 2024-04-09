import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from data_import import training_dataset, player_ids 
from data_processing import calculate_team_statistic, playeravg_features, apply_playerfeatures

def create_testdataset(team1_players, team2_players, player_id_dict):
    # Create DataFrame for the match
    match_data = pd.DataFrame(columns=['Match ID'] + [f'Team {i+1} P{j+1}' for i in range(2) for j in range(8)] +
                                        ['Team 1 Goals', 'Team 2 Goals', 'Team 1 Result', 'Team 2 Result'])

    # Assign players for Team 1
    for team_name, players in zip(['Team 1', 'Team 2'], [team1_players, team2_players]):
        for i, player in enumerate(players):
            match_data.at[0, f'{team_name} P{i+1}'] = player

    match_data.fillna(0, inplace=True)

    try:
        # Replace player names with player IDs
        for column in match_data.columns:
            if column.startswith('Team 1 P') or column.startswith('Team 2 P'):
                match_data[column] = match_data[column].apply(lambda x: player_id_dict[x])

    except Exception as e:
        raise RuntimeError(f"An error occurred while replacing player names with player IDs: {str(e)} is not in player_id_dict")
    
    return match_data

#Who are the teams you want to run through the model?
team1_players = ['Waq', 'Jamie', 'Saeed', 'Sam', 'Jason', 'Yusuf', 'Carlos', 'Mo.O']
team2_players = ['Saqi', 'Jake', 'Satpal', 'Ashley', 'Shyam', 'Riz', 'Rahul', 'Saj']

player_id_dict = dict(zip(player_ids['player name'], player_ids['player_ID']))

test_dataset = create_testdataset(team1_players, team2_players, player_id_dict)

win_percentage_dict, avg_GS_dict, avg_GC_dict = playeravg_features(training_dataset)

clf = RandomForestClassifier()
reg_team1 = RandomForestRegressor()
reg_team2 = RandomForestRegressor()

def training_clf(training_dataset):
    df = apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, training_dataset)
    df = df.select_dtypes(exclude=['object']) 
    df["Team 1 Win"] = df["Team 1 Result"].apply(lambda x: 1 if x == 1 else 0)
    x_clf = df.drop(["Team 1 Result", "Team 2 Result", "Team 1 Win", "Team 1 Goals", "Team 2 Goals"], axis=1)
    y_clf = df["Team 1 Win"]
    clf.fit(x_clf, y_clf)
    return

def training_rfg(training_dataset):
    df = apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, training_dataset)
    df = df.select_dtypes(exclude=['object']) 
    x_reg = df.drop(["Team 1 Result", "Team 2 Result", "Team 1 Goals", "Team 2 Goals"], axis=1)
    y_reg_team1 = df["Team 1 Goals"]
    y_reg_team2 = df["Team 2 Goals"] 
    reg_team1.fit(x_reg, y_reg_team1)
    reg_team2.fit(x_reg, y_reg_team2)
    return

def predict_scores(test_dataset):
    df = apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, test_dataset)
    df = df.select_dtypes(exclude=['object'])
    x_test = df.drop(["Team 1 Result", "Team 2 Result", "Team 1 Goals", "Team 2 Goals"], axis=1)
    team1_goals = reg_team1.predict(x_test)
    team2_goals = reg_team2.predict(x_test)
    team1_win = clf.predict(x_test)
    return team1_goals, team2_goals, team1_win

training_clf(training_dataset)
training_rfg(training_dataset)
team1_goals, team2_goals, team1_win = predict_scores(test_dataset)

# Print predicted goals
print("Predicted Goals:")
print("Team 1:", team1_goals)
print("Team 2:", team2_goals)
print("\nPredicted Results:")
if team1_win == 1:
    print("Team 1")
else:
    print("Team 2")


