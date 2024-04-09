import pandas as pd
from collections import defaultdict

#Calculate additional features prior to model training

def calculate_team_statistic(player_ids, stat_dict):
    valid_player_ids = [player_id for player_id in player_ids if player_id != 0]
    if valid_player_ids:
        return sum(stat_dict.get(player_id, 0) for player_id in valid_player_ids) / len(valid_player_ids)
    else:
        return 0
    
def playeravg_features(training_set):
    player_stats = {}
    df = training_set

    # Iterate over each match to collect player statistics
    for index, row in df.iterrows():
        # Extract player IDs for both teams
        team1_players = [player_id for player_id in row[['Team 1 P1', 'Team 1 P2', 'Team 1 P3', 'Team 1 P4', 'Team 1 P5', 'Team 1 P6', 'Team 1 P7', 'Team 1 P8']] if player_id != 0]
        team2_players = [player_id for player_id in row[['Team 2 P1', 'Team 2 P2', 'Team 2 P3', 'Team 2 P4', 'Team 2 P5', 'Team 2 P6', 'Team 2 P7', 'Team 2 P8']] if player_id != 0]

        for team, players in [('Team 1', team1_players), ('Team 2', team2_players)]:
            for player_id in players:
                player_stats[player_id] = player_stats.get(player_id, {"matches_played": 0, "matches_won": 0, "total_GS": 0, "total_GC": 0})
                player_stats[player_id]["matches_played"] += 1
                player_stats[player_id]["total_GS"] += row[f'{team} Goals']
                player_stats[player_id]["total_GC"] += row[f'Team 1 Goals' if team == 'Team 2' else 'Team 2 Goals']
                if row[f'{team} Result'] == 1:
                    player_stats[player_id]["matches_won"] += 1
    
    # Calculate win percentage, average goals, and average goals conceded for each player
    win_percentage_dict = {}
    avg_GS_dict = {}
    avg_GC_dict = {}

    for player_id, stats in player_stats.items():
        win_percentage_dict[player_id] = (stats["matches_won"] / stats["matches_played"]) * 100 if stats["matches_played"] > 0 else 0
        avg_GS_dict[player_id] = stats["total_GS"] / stats["matches_played"]
        avg_GC_dict[player_id] = stats["total_GC"] / stats["matches_played"]
    
    return win_percentage_dict, avg_GS_dict, avg_GC_dict

def apply_playerfeatures (win_percentage_dict, avg_GS_dict, avg_GC_dict, dataset):
    df = dataset
    for team_num in range(1, 3):
        team_label = f'team{team_num}'
        team_label2 = f'Team {team_num}'
        df[f'{team_label}_win_percentage'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], win_percentage_dict), axis=1)
        df[f'{team_label}_avg_goals'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], avg_GS_dict), axis=1)
        df[f'{team_label}_avg_goalsconceded'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label2} P{i+1}' for i in range(8)]], avg_GC_dict), axis=1)

    columns_to_convert = ['team1_win_percentage', 'team2_win_percentage', 'team1_avg_goals', 'team2_avg_goals', 'team1_avg_goalsconceded', 'team2_avg_goalsconceded']
    df[columns_to_convert] = df[columns_to_convert].astype('float64')

    return df