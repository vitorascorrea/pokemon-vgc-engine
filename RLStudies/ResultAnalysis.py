import pickle
import pandas as pd
import matplotlib.pyplot as plt

from RLStudies.Utils import Test

def parse_matches_results(matches_array, match_id):
    # 'DQN vs random: {-1: 0, 0: 7, 1: 93} {-1: 0, 0: 0, 1: 100}'
    match = None
    for match_ in matches_array:
        if match_id in match_:
            match = match_
            break

    splitted_1 = match.split(': {')
    splitted_2 = splitted_1[1].split('} {')
    splitted_2[0] = '{' + splitted_2[0] + '}'
    splitted_2[1] = '{' + splitted_2[1]

    team_a_vs_b = eval(splitted_2[0])
    team_b_vs_a = eval(splitted_2[1])

    victories = (team_a_vs_b[0] + team_b_vs_a[0]) / 2
    defeats = (team_a_vs_b[1] + team_b_vs_a[1]) / 2

    return victories - defeats

if __name__ == "__main__":
    dqn_no_clip_results = None
    dqn_with_clip_results = None
    ddqn_no_clip_results = None
    ddqn_with_clip_results = None

    with open('./RLStudies/results/DQN_no_clip.pickle', 'rb') as handle:
        dqn_no_clip_results = pickle.load(handle)

    with open('./RLStudies/results/dqn_with_clip.pickle', 'rb') as handle:
        dqn_with_clip_results = pickle.load(handle)

    with open('./RLStudies/results/ddqn_no_clip.pickle', 'rb') as handle:
        ddqn_no_clip_results = pickle.load(handle)

    with open('./RLStudies/results/ddqn_with_clip.pickle', 'rb') as handle:
        ddqn_with_clip_results = pickle.load(handle)

    # loss function
    # data = {
    #     'Episode': [item[0] for item in dqn_no_clip_results.loss_history[4:]],
    #     'Loss_DQN_no_clip': [item[1] for item in dqn_no_clip_results.loss_history[4:]],
    #     'Loss_DQN_with_clip': [item[1] for item in dqn_with_clip_results.loss_history[4:]],
    #     'Loss_DDQN_no_clip': [item[1] for item in ddqn_no_clip_results.loss_history[4:]],
    #     'Loss_DDQN_with_clip': [item[1] for item in ddqn_with_clip_results.loss_history[4:]],
    # }

    # df = pd.DataFrame(data)
    # print(df)
    # plt.plot('Episode', 'Loss_DQN_no_clip', data=df, marker='', color='blue')
    # plt.plot('Episode', 'Loss_DQN_with_clip', data=df, marker='', color='green')
    # plt.plot('Episode', 'Loss_DDQN_no_clip', data=df, marker='', color='red')
    # plt.plot('Episode', 'Loss_DDQN_with_clip', data=df, marker='', color='olive')

    # plt.legend()
    # plt.show()

    # victories
    data = {
        'Episode': [item[0] for item in dqn_no_clip_results.matches_history],
        # 'DQN_vs_random_no_clip': [parse_matches_results(item[1], 'DQN vs random') for item in dqn_no_clip_results.matches_history],
        # 'DQN_vs_random_with_clip': [parse_matches_results(item[1], 'DQN vs random') for item in dqn_with_clip_results.matches_history],
        # 'DDQN_vs_random_no_clip': [parse_matches_results(item[1], 'DQN vs random') for item in ddqn_no_clip_results.matches_history],
        # 'DDQN_vs_random_with_clip': [parse_matches_results(item[1], 'DQN vs random') for item in ddqn_with_clip_results.matches_history],
        'DQN_vs_greedy_no_clip': [parse_matches_results(item[1], 'DQN vs greedy') for item in dqn_no_clip_results.matches_history],
        'DQN_vs_greedy_with_clip': [parse_matches_results(item[1], 'DQN vs greedy') for item in dqn_with_clip_results.matches_history],
        'DDQN_vs_greedy_no_clip': [parse_matches_results(item[1], 'DQN vs greedy') for item in ddqn_no_clip_results.matches_history],
        'DDQN_vs_greedy_with_clip': [parse_matches_results(item[1], 'DQN vs greedy') for item in ddqn_with_clip_results.matches_history],
    }

    df = pd.DataFrame(data)
    print(df.to_latex(index=False))
    # print(df[df['DQN_vs_random_no_clip'] > -100]['DQN_vs_random_no_clip'].mean())
    # print(df[df['DQN_vs_random_with_clip'] > -100]['DQN_vs_random_with_clip'].mean())
    # print(df[df['DDQN_vs_random_no_clip'] > -100]['DDQN_vs_random_no_clip'].mean())
    # print(df[df['DDQN_vs_random_with_clip'] > -100]['DDQN_vs_random_with_clip'].mean())
    print(df[df['DQN_vs_greedy_no_clip'] > -100]['DQN_vs_greedy_no_clip'].mean())
    print(df[df['DQN_vs_greedy_with_clip'] > -100]['DQN_vs_greedy_with_clip'].mean())
    print(df[df['DDQN_vs_greedy_no_clip'] > -100]['DDQN_vs_greedy_no_clip'].mean())
    print(df[df['DDQN_vs_greedy_with_clip'] > -100]['DDQN_vs_greedy_with_clip'].mean())
