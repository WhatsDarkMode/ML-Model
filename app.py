from flask import Flask, render_template, request
import pandas as pd
import data_processing as f1
import model_training as f2

app = Flask(__name__)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

training_dataset = pd.read_csv("ML_Data_Set.csv")
player_ids = pd.read_csv("Player keys.csv")
player_list = player_ids['player name']

@app.route('/')
def home():
    return render_template('index.html', player_list = player_list)

@app.route('/submit_teams', methods=['POST'])
def submit_teams():
    team1_players = [request.form[f'team1_player{i}'] for i in range(1, 8)]
    team2_players = [request.form[f'team2_player{i}'] for i in range(1, 8)]
    print(team1_players)
    print(team2_players)

    test_dataset = f2.create_testdataset(team1_players, team2_players, player_ids)

    processed_training_dataset, processed_testing_dataset = f1.process_dataset(training_dataset, test_dataset)

    team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob = f2.train_test_models(processed_training_dataset, processed_testing_dataset)

    f2.print_results(team1_goals, team2_goals, team1_win, team1_win_prob, team2_win_prob)

    return render_template('results.html', team1_goals=team1_goals[0], team2_goals=team2_goals[0], team1_win_prob=team1_win_prob[0], team2_win_prob=team2_win_prob[0], team1_win=team1_win)

context = ('./ssl/client-cert.pem','./ssl/client-key.pem')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, ssl_context = context)
