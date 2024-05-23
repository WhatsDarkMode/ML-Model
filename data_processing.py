import pandas as pd
from collections import defaultdict

def calculate_team_statistic(player_ids, stat_dict):
    valid_player_ids = [player_id for player_id in player_ids if player_id != 0]
    if valid_player_ids:
        return sum(stat_dict.get(player_id, 0) for player_id in valid_player_ids) / len(valid_player_ids)
    else:
        return 0

def playeravg_features(default_dataset):
    player_stats = {}
    df = default_dataset

    for index, row in df.iterrows():
        team1_players = [row[f'Team 1 P{i}'] for i in range(1, 9) if row[f'Team 1 P{i}'] != 0]
        team2_players = [row[f'Team 2 P{i}'] for i in range(1, 9) if row[f'Team 2 P{i}'] != 0]

        for team, players in [('Team 1', team1_players), ('Team 2', team2_players)]:
            for player_id in players:
                if player_id not in player_stats:
                    player_stats[player_id] = {"matches_played": 0, "matches_won": 0, "total_GS": 0, "total_GC": 0}
                player_stats[player_id]["matches_played"] += 1
                player_stats[player_id]["total_GS"] += row[f'{team} Goals']
                player_stats[player_id]["total_GC"] += row[f'Team 1 Goals' if team == 'Team 2' else 'Team 2 Goals']
                if row[f'{team} Result'] == 1:
                    player_stats[player_id]["matches_won"] += 1
    
    win_percentage_dict = {}
    avg_GS_dict = {}
    avg_GC_dict = {}

    for player_id, stats in player_stats.items():
        win_percentage_dict[player_id] = (stats["matches_won"] / stats["matches_played"]) * 100 if stats["matches_played"] > 0 else 0
        avg_GS_dict[player_id] = stats["total_GS"] / stats["matches_played"]
        avg_GC_dict[player_id] = stats["total_GC"] / stats["matches_played"]
    
    return win_percentage_dict, avg_GS_dict, avg_GC_dict

#This feature calculates duo stats
def calculate_duo_stats(training_dataset):
    # Initialize dictionaries to store player combinations and win rates
    player_combinations = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'goals': 0, 'goals_conceded': 0, 'matches': 0}))
    win_rates = defaultdict(lambda: defaultdict(float))

    # Iterate over each match to collect player combinations and statistics
    for _, row1 in training_dataset.iterrows():
        team1_players = [int(row1[f'Team 1 P{i}']) for i in range(1, 9) if row1[f'Team 1 P{i}'] != 0]
        team2_players = [int(row1[f'Team 2 P{i}']) for i in range(1, 9) if row1[f'Team 2 P{i}'] != 0]

        for player1 in team1_players:
            for player2 in team2_players:
                player_combinations[player1][player2]['matches'] += 1
                player_combinations[player2][player1]['matches'] += 1

                if row1['Team 1 Result'] == 1:
                    player_combinations[player1][player2]['wins'] += 1
                elif row1['Team 2 Result'] == 1:
                    player_combinations[player2][player1]['wins'] += 1

                goals_for = int(row1['Team 1 Goals'])
                goals_against = int(row1['Team 2 Goals'])

                player_combinations[player1][player2]['goals'] += goals_for
                player_combinations[player2][player1]['goals'] += goals_for
                player_combinations[player1][player2]['goals_conceded'] += goals_against
                player_combinations[player2][player1]['goals_conceded'] += goals_against

    # Calculate win rates from player combinations
    for player, teammates in player_combinations.items():
        for teammate, stats in teammates.items():
            if stats['matches'] > 0:
                wins_together = stats['wins']
                common_matches = stats['matches']
                win_rates[player][teammate] = wins_together / common_matches * 100

    return player_combinations, win_rates

def calculate_duo_win_rate(player_ids, win_rates):
    valid_player_ids = [player_id for player_id in player_ids if player_id != 0]

    duo_win_rates = []
    num_players = len(valid_player_ids)

    for i in range(num_players):
        for j in range(i + 1, num_players):
            player1 = valid_player_ids[i]
            player2 = valid_player_ids[j]
            if player1 in win_rates and player2 in win_rates[player1]:
                duo_win_rates.append(win_rates[player1][player2])

    if duo_win_rates:
        return sum(duo_win_rates) / len(duo_win_rates)
    else:
        return 0

def apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, win_rates, default_dataset):
    df = default_dataset
    for team_num in range(1, 3):
        team_label = f'team{team_num}'
        team_label2 = f'Team {team_num}'
        df[f'{team_label}_win_percentage'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], win_percentage_dict), axis=1)
        df[f'{team_label}_avg_goals'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], avg_GS_dict), axis=1)
        df[f'{team_label}_avg_goalsconceded'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], avg_GC_dict), axis=1)
        df[f'{team_label}_duo_win_rate'] = df.apply(lambda row: calculate_duo_win_rate(row[[f'{team_label2} P{i+1}' for i in range(8)]], win_rates), axis=1)

    columns_to_convert = ['team1_win_percentage', 'team2_win_percentage', 'team1_avg_goals', 'team2_avg_goals', 'team1_avg_goalsconceded', 'team2_avg_goalsconceded', 'team1_duo_win_rate', 'team2_duo_win_rate']
    df[columns_to_convert] = df[columns_to_convert].astype('float64')

    return df

def process_dataset(training_dataset, test_dataset):
    win_percentage_dict, avg_GS_dict, avg_GC_dict = playeravg_features(training_dataset)
    player_combinations, win_rates = calculate_duo_stats(training_dataset)
    processed_training_dataset = apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, win_rates, training_dataset)
    processed_test_dataset = apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, win_rates, test_dataset)
    return processed_training_dataset, processed_test_dataset






































# import pandas as pd
# from collections import defaultdict

# # To calculate the teams win percentage, avg goals scored, avg goals conceded
# def calculate_team_statistic(player_ids, stat_dict):
#     valid_player_ids = [player_id for player_id in player_ids if player_id != 0]
#     if valid_player_ids:
#         return sum(stat_dict.get(player_id, 0) for player_id in valid_player_ids) / len(valid_player_ids)
#     else:
#         return 0

# # Calculate the average goals scored, goals conceded and win percentage of each individual player
# def playeravg_features(default_dataset):
#     player_stats = {}
#     df = default_dataset

#     # Iterate over each match to collect player statistics
#     for index, row in df.iterrows():
#         # Extract player IDs for both teams
#         team1_players = [player_id for player_id in row[['Team 1 P1', 'Team 1 P2', 'Team 1 P3', 'Team 1 P4', 'Team 1 P5', 'Team 1 P6', 'Team 1 P7', 'Team 1 P8']] if player_id != 0]
#         team2_players = [player_id for player_id in row[['Team 2 P1', 'Team 2 P2', 'Team 2 P3', 'Team 2 P4', 'Team 2 P5', 'Team 2 P6', 'Team 2 P7', 'Team 2 P8']] if player_id != 0]

#         for team, players in [('Team 1', team1_players), ('Team 2', team2_players)]:
#             for player_id in players:
#                 player_stats[player_id] = player_stats.get(player_id, {"matches_played": 0, "matches_won": 0, "total_GS": 0, "total_GC": 0})
#                 player_stats[player_id]["matches_played"] += 1
#                 player_stats[player_id]["total_GS"] += row[f'{team} Goals']
#                 player_stats[player_id]["total_GC"] += row[f'Team 1 Goals' if team == 'Team 2' else 'Team 2 Goals']
#                 if row[f'{team} Result'] == 1:
#                     player_stats[player_id]["matches_won"] += 1
    
#     # Calculate win percentage, average goals, and average goals conceded for each player
#     win_percentage_dict = {}
#     avg_GS_dict = {}
#     avg_GC_dict = {}

#     for player_id, stats in player_stats.items():
#         win_percentage_dict[player_id] = (stats["matches_won"] / stats["matches_played"]) * 100 if stats["matches_played"] > 0 else 0
#         avg_GS_dict[player_id] = stats["total_GS"] / stats["matches_played"]
#         avg_GC_dict[player_id] = stats["total_GC"] / stats["matches_played"]
    
#     return win_percentage_dict, avg_GS_dict, avg_GC_dict


# #This feature calculates duo stats
# def calculate_duo_stats(default_dataset):
#     # Initialize dictionaries to store player combinations and win rates
#     player_combinations = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'goals': 0, 'goals_conceded': 0, 'matches': 0}))
#     win_rates = defaultdict(lambda: defaultdict(float))

#     # Iterate over each match to collect player combinations and statistics
#     for _, row1 in default_dataset.iterrows():
#         team1_players = [int(row1[f'Team 1 P{i}']) for i in range(1, 9) if row1[f'Team 1 P{i}'] != 0]
#         team2_players = [int(row1[f'Team 2 P{i}']) for i in range(1, 9) if row1[f'Team 2 P{i}'] != 0]

#         for player1 in team1_players:
#             for player2 in team2_players:
#                 player_combinations[player1][player2]['matches'] += 1
#                 player_combinations[player2][player1]['matches'] += 1

#                 if row1['Team 1 Result'] == 1:
#                     player_combinations[player1][player2]['wins'] += 1
#                 elif row1['Team 2 Result'] == 1:
#                     player_combinations[player2][player1]['wins'] += 1

#                 goals_for = int(row1['Team 1 Goals'])
#                 goals_against = int(row1['Team 2 Goals'])

#                 player_combinations[player1][player2]['goals'] += goals_for
#                 player_combinations[player2][player1]['goals'] += goals_for
#                 player_combinations[player1][player2]['goals_conceded'] += goals_against
#                 player_combinations[player2][player1]['goals_conceded'] += goals_against

#     # Calculate win rates from player combinations
#     for player, teammates in player_combinations.items():
#         for teammate, stats in teammates.items():
#             if stats['matches'] > 0:
#                 wins_together = stats['wins']
#                 common_matches = stats['matches']
#                 win_rates[player][teammate] = wins_together / common_matches * 100

#     return player_combinations, win_rates

# # Applies the functions to the datasets
# def apply_playerfeatures (win_percentage_dict, avg_GS_dict, avg_GC_dict, default_dataset):
#     df = default_dataset
#     for team_num in range(1, 3):
#         team_label = f'team{team_num}'
#         team_label2 = f'Team {team_num}'
#         df[f'{team_label}_win_percentage'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], win_percentage_dict), axis=1)
#         df[f'{team_label}_avg_goals'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], avg_GS_dict), axis=1)
#         df[f'{team_label}_avg_goalsconceded'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], avg_GC_dict), axis=1)

#     columns_to_convert = ['team1_win_percentage', 'team2_win_percentage', 'team1_avg_goals', 'team2_avg_goals', 'team1_avg_goalsconceded', 'team2_avg_goalsconceded']
#     df[columns_to_convert] = df[columns_to_convert].astype('float64')

#     return df

# # One process_dataset function that simplifies calling all the other processing functions
# def process_dataset(training_dataset, test_dataset):
#     win_percentage_dict, avg_GS_dict, avg_GC_dict = playeravg_features(training_dataset)
#     #player_combinations, win_rates = calculate_duo_stats(default_dataset) # currently not built-in to use this
#     processed_training_dataset = apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, training_dataset)
#     processed_test_dataset = apply_playerfeatures(win_percentage_dict, avg_GS_dict, avg_GC_dict, test_dataset)
#     return processed_training_dataset, processed_test_dataset