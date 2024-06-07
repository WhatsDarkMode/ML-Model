import pandas as pd
import data_processing as f1
import model_training as f2

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

training_dataset = pd.read_csv("ML_Data_Set.csv")
player_ids = pd.read_csv("Player keys.csv")

team1_players = ['Mike', 'Jake', 'Rob', 'Rahul', 'Paddy', 'Shyam', 'Mani']
team2_players = ['Kal', 'Jamie', 'Sam', 'Saeed', 'Kam', 'Omar', 'Waq']

test_dataset = f2.create_testdataset(team1_players, team2_players, player_ids)

processed_training_dataset, processed_testing_dataset = f1.process_dataset(training_dataset, test_dataset)

print("below is training dataset")
print(processed_training_dataset)

print("below is the test data set")
print(processed_testing_dataset)

team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob = f2.train_test_models(processed_training_dataset, processed_testing_dataset)

f2.print_results(team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob)

































# import pandas as pd
# from tabulate import tabulate
# import data_processing as f1
# import model_training as f2

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

# # Read in both datasets
# training_dataset = pd.read_csv("ML_Data_Set.csv")
# player_ids = pd.read_csv("Player keys.csv")

# # Who are the teams you want to run through the model?
# team1_players = ['Sam', 'Jake', 'Manpreet', 'Jason', 'Mike', 'Shyam', 'Paddy']
# team2_players = ['Kal', 'Jamie', 'Satpal', 'Kal', 'Waq', 'Omar', 'Rahul']

# # Create test dataset
# test_dataset = f2.create_testdataset(team1_players, team2_players, player_ids)

# # Transform 
# processed_training_dataset, processed_testing_dataset = f1.process_dataset(training_dataset, test_dataset)

# # Run prediction
# team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob, draw = f2.train_test_models(processed_training_dataset, processed_testing_dataset)

# # Print Results
# f2.print_results(team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob, draw)