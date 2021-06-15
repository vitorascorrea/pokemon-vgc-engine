import random

from framework.process.BattleEngine import PkmBattleEnv
from framework.util.PkmTeamGenerators import RandomGenerator
from framework.DataConstants import DEFAULT_TEAM_SIZE, MAX_TEAM_SIZE


def build_random_teams(num_of_teams=2):
    rg = RandomGenerator()
    teams = []

    for _ in range(num_of_teams):
        team = rg.get_team().get_battle_team(random.sample(range(MAX_TEAM_SIZE), DEFAULT_TEAM_SIZE))
        teams.append(team)

    return teams


def test(team0=None, team1=None, random_teams=True, opponent_action_fn=None, player_action_fn=None):
    if random_teams or not team0 or not team1:
        team0, team1 = build_random_teams()

    env = PkmBattleEnv(debug=True, teams=[team0, team1])
    state = env.reset()
    state_view = env.game_state_view

    done = False
    turns = 0

    while not done:
        turns += 1
        if turns > 500:
            break

        player_action = player_action_fn(state[0], state_view[0])
        opponent_action = opponent_action_fn(state[1], state_view[1])

        action = [player_action, opponent_action]

        state, reward, done, state_view = env.step(action)

    return env.winner


def test_batch(num_tests=100, team0=None, team1=None, random_teams=False, opponent_action_fn=None, player_action_fn=None):
    winners = {
        -1: 0,
        0: 0,
        1: 0
    }

    for _ in range(num_tests):
        winner = test(team0=team0, team1=team1, random_teams=random_teams, opponent_action_fn=opponent_action_fn, player_action_fn=player_action_fn)
        winners[winner] += 1

    return winners
