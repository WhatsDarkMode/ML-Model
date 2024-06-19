import pandas as pd
from collections import defaultdict
from itertools import combinations

def calculate_team_statistic(player_ids, player_stats_per_game, stat_key):
    valid_player_ids = [player_id for player_id in player_ids if player_id != 0] 
    if valid_player_ids:
        return sum(player_stats_per_game.get(player_id, {}).get(stat_key, 0) for player_id in valid_player_ids) / len(valid_player_ids)
    else:
        return 0

def playeravg_features(training_dataset):
    """
    playeravg_features function

    This function takes a 'training_dataset', one that has not been processed or altered yet.
    It works out the players total stats across the training_dataset for several parameters:
    Matches played
    Wins
    Goals scored (by the player's team)
    Goals conceded (by the player's team)

    It returns a single dictionary containing 3 keys; win %, avg total goals scored, avg total goals conceded
    """
    player_stats = defaultdict(lambda: {'matches_played': 0, 'matches_won': 0, 'total_gs': 0, 'total_gc': 0})
    df = training_dataset
    
    for index, row in df.iterrows():
        team1_players = [row[f'Team 1 P{i}'] for i in range(1,9) if row[f'Team 1 P{i}'] != 0] # gets team 1 players
        team2_players = [row[f'Team 2 P{i}'] for i in range(1,9) if row[f'Team 2 P{i}'] != 0] # gets team 2 players
    
        for team, players in [('Team 1', team1_players), ('Team 2', team2_players)]: #this creates a alist of tuples, each tuple is either "Team 1" or "Team 2" as value one, and the respective list of players as value 2
            for player_id in players:
                player_stats[player_id]['matches_played'] += 1
                player_stats[player_id]['total_gs'] += row[f'{team} Goals'] # using format string to access a specific value from a column in the df
                player_stats[player_id]['total_gc'] += row[f'Team 1 Goals' if team == 'Team 2' else 'Team 2 Goals']
                player_stats[player_id]['matches_won'] += 1 if row[f'{team} Result'] == 1 else 0 # this is a conditional expression

    player_stats_per_game = defaultdict(lambda: {'win_percentage': 0, 'avg_gs': 0, 'avg_gc': 0})

    for player_id, stats in player_stats.items():
        if stats['matches_played'] > 0:
            player_stats_per_game[player_id]['win_percentage'] = (stats['matches_won'] / stats['matches_played']) * 100
            player_stats_per_game[player_id]['avg_gs'] = stats['total_gs'] / stats['matches_played']
            player_stats_per_game[player_id]['avg_gc'] = stats['total_gc'] / stats['matches_played']          

    return player_stats_per_game

#This feature calculates duo stats
def calculate_duo_stats(training_dataset):
    """
    calculate_duo_stats function

    This function takes the training_dataset and parses through each row. As it does it calculates each players stats with their other teammates.
    It works out every players goals, goals_conceded, matches, wins with every other player
    it returns two dictionaries, one containing all the players and each others stats together. 
    """
    player_combinations = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'goals': 0, 'goals_conceded': 0, 'matches': 0, 'win_rate': 0}))
    df = training_dataset

    for index, row in df.iterrows():
        team1_players = [row[f'Team 1 P{i}'] for i in range(1,9) if row[f'Team 1 P{i}'] != 0] # gets team 1 players
        team2_players = [row[f'Team 2 P{i}'] for i in range(1,9) if row[f'Team 2 P{i}'] != 0] # gets team 2 players         
        # calculate team 1's players combo for that match
        for team1_player in team1_players:
            for team1_teammate in team1_players:
                if team1_player != team1_teammate:
                    player_combinations[team1_player][team1_teammate]['matches'] += 1
                    player_combinations[team1_player][team1_teammate]['wins'] += 1 if row['Team 1 Result'] == 1 else 0
                    player_combinations[team1_player][team1_teammate]['goals'] += row['Team 1 Goals']
                    player_combinations[team1_player][team1_teammate]['goals_conceded'] += row['Team 2 Goals']
        # calculate team 2's players combo for that match
        for team2_player in team2_players:
            for team2_teammate in team2_players:
                if team2_player != team2_teammate:
                    player_combinations[team2_player][team2_teammate]['matches'] += 1          
                    player_combinations[team2_player][team2_teammate]['wins'] += 1 if row['Team 2 Result'] == 1 else 0
                    player_combinations[team2_player][team2_teammate]['goals'] += row['Team 2 Goals']
                    player_combinations[team2_player][team2_teammate]['goals_conceded'] += row['Team 1 Goals']

    # Calculate win rates from player combinations
    for player, teammates in player_combinations.items():
        for teammate, stats in teammates.items():
            if stats['matches'] > 0:
                wins_together = stats['wins']
                common_matches = stats['matches']
                player_combinations[player][teammate]['win_rate'] = wins_together / common_matches * 100
    
    return player_combinations

def calculate_duo_statistic(team_player_ids, player_combinations):
    total_win_rate = 0
    pair_count = 0

    for player1, player2 in combinations(team_player_ids, 2):
        total_win_rate += player_combinations[player1][player2].get('win_rate', 0)
        pair_count += 1

    if pair_count == 0:
        return 0

    return total_win_rate / pair_count

def apply_playerfeatures(player_stats_per_game, player_combinations, default_dataset):
    df = default_dataset
    
    for team_num in range(1,3):
        team_label_no_space = f'team{team_num}'
        team_label_with_space = f'Team {team_num}'
        df[f'{team_label_no_space}_win_percentage'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label_with_space} P{i+1}' for i in range(8)]], player_stats_per_game, 'win_percentage'), axis = 1)
        df[f'{team_label_no_space}_avg_goals'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label_with_space} P{i+1}' for i in range(8)]], player_stats_per_game, 'avg_gs'), axis = 1)
        df[f'{team_label_no_space}_avg_goalsconceded'] = df.apply(lambda row: calculate_team_statistic(row[[f'{team_label_with_space} P{i+1}' for i in range(8)]], player_stats_per_game, 'avg_gc'), axis = 1)
        df[f'{team_label_no_space}_avg_duo_winrate'] = df.apply(lambda row: calculate_duo_statistic(row[[f'{team_label_with_space} P{i+1}' for i in range(8)]], player_combinations), axis = 1)          
    
    columns_to_convert = ['team1_win_percentage', 'team2_win_percentage', 'team1_avg_goals', 'team2_avg_goals', 'team1_avg_goalsconceded', 'team2_avg_goalsconceded', 'team1_avg_duo_winrate', 'team2_avg_duo_winrate']
    df[columns_to_convert] = df[columns_to_convert].astype('float64')

    return df

def process_dataset(training_dataset, test_dataset):
    player_stats_per_game = playeravg_features(training_dataset)
    player_combinations = calculate_duo_stats(training_dataset)
    processed_training_dataset = apply_playerfeatures(player_stats_per_game, player_combinations, training_dataset)
    processed_test_dataset = apply_playerfeatures(player_stats_per_game, player_combinations, test_dataset)
    return processed_training_dataset, processed_test_dataset

