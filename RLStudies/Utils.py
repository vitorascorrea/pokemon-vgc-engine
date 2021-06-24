import random
import torch

from collections import namedtuple
from competitor.greedy.Greedy import GreedyBattlePolicy
from competitor.random.Random import RandomBattlePolicy
from RLStudies.Tester import build_random_teams, test_batch
from framework.behaviour.BattlePolicies import estimate_damage
from framework.DataTypes import PkmStat
from framework.DataConstants import DEFAULT_PARTY_SIZE, DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS, TYPE_CHART_MULTIPLIER

torch.manual_seed(0)

GAMMA = 0.999
PK_TEAM_ACTIONS = DEFAULT_PKM_N_MOVES + DEFAULT_PARTY_SIZE
TransitionIterator = namedtuple('TransitionIterator',
                                ('state', 'action', 'next_state', 'reward', 'state_action',  'acc_reward'))


class Test():
    def __init__(self, name, double_dqn=False, clip_reward=False) -> None:
        self.name = name
        self.loss_history = []
        self.matches_history = []
        self.double_dqn = double_dqn
        self.clip_reward = clip_reward


class Transition():
    def __init__(self, state, action, next_state, reward, state_action, acc_reward) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.state_action = state_action
        self.acc_reward = acc_reward

    def __iter__(self):
        return iter(TransitionIterator(
            self.state,
            self.action,
            self.next_state,
            self.reward,
            self.state_action,
            self.acc_reward
        ))


def calculate_one_hot(index, length):
    return [1 if index == i else 0 for i in range(length)]


def build_state_representation(state_view):
    # check weather condition
    weather = state_view.weather_condition

    # get my team
    my_team = state_view.get_team_view(0)
    my_active = my_team.active_pkm_view
    my_active_type = my_active.type
    my_party = [my_team.get_party_pkm_view(i) for i in range(DEFAULT_PARTY_SIZE)]
    my_active_moves = [my_active.get_move_view(i) for i in range(DEFAULT_PKM_N_MOVES)]
    my_attack_stage = my_team.get_stage(PkmStat.ATTACK)

    # get opp team
    opp_team = state_view.get_team_view(1)
    opp_active = opp_team.active_pkm_view
    opp_active_type = opp_active.type
    opp_active_hp = opp_active.hp
    opp_defense_stage = opp_team.get_stage(PkmStat.DEFENSE)

    # get best move
    move_choices = [0] * DEFAULT_PKM_N_MOVES
    switch_choices = [0] * DEFAULT_PARTY_SIZE

    for i, move in enumerate(my_active_moves):
        move_damage = estimate_damage(move.type, my_active_type, move.power, opp_active_type, my_attack_stage, opp_defense_stage, weather)
        move_choices[i] = move_damage if move.pp != 0 else 0

    for j, pkm in enumerate(my_party):
        effectiveness_party = TYPE_CHART_MULTIPLIER[pkm.type][opp_active_type]
        switch_choices[j] = effectiveness_party if pkm.hp != 0 else 0

    my_hp_percentage = (my_active.hp / MAX_HIT_POINTS) * 100
    opp_hp_percentage = (opp_active_hp / MAX_HIT_POINTS) * 100

    raw_state = [my_hp_percentage, opp_hp_percentage] + move_choices + switch_choices
    all_values_sum = sum(raw_state)
    std_state = [i / all_values_sum for i in raw_state]

    return std_state


def select_player_action_from_q_value(state, q_value_nn):
    action_index = 0
    action_value = float('-inf')
    q_value_nn.eval()

    with torch.no_grad():
        for i in range(PK_TEAM_ACTIONS):
            state_tensor = torch.as_tensor([state + calculate_one_hot(i, PK_TEAM_ACTIONS)])
            predicted_q_value = q_value_nn(state_tensor)
            q_value = predicted_q_value.item()

            if q_value > action_value:
                action_value = q_value
                action_index = i

    return action_index


def run_episode(env, player_action_fn, opponent_action_fn, step_fn=None, compute_reward_fn=None, last_n_turns=None, store_opponent_memory=True):
    previous_state = None
    previous_state_view = None
    state = env.reset()
    state_view = env.game_state_view

    done = False
    turns = 0

    episode_player_memory = []
    episode_opponent_memory = []

    while not done:
        turns += 1
        if turns > 200:
            break

        player_action = player_action_fn(state[0], state_view[0])
        opponent_action = opponent_action_fn(state[1], state_view[1])

        action = [player_action, opponent_action]

        previous_state = state
        previous_state_view = state_view

        state, reward, done, state_view = env.step(action)

        if compute_reward_fn:
            reward[0] = compute_reward_fn(state_view=state_view[0], winner=env.winner == 0 if done else None)
            reward[1] = compute_reward_fn(state_view=state_view[1], winner=env.winner == 1 if done else None)

        previous_state[0] = build_state_representation(previous_state_view[0])
        previous_state[1] = build_state_representation(previous_state_view[1])

        state[0] = build_state_representation(state_view[0])
        state[1] = build_state_representation(state_view[1])

        # Run some step function if necessary
        if step_fn:
            step_fn(turns)

        # Store the transition in memory
        episode_player_memory.append(
            Transition(
                state=previous_state[0],
                action=player_action,
                next_state=state[0],
                reward=reward[0],
                state_action=previous_state[0] + calculate_one_hot(player_action, PK_TEAM_ACTIONS),
                acc_reward=0
            )
        )

        if store_opponent_memory:
            episode_opponent_memory.append(
                Transition(
                    state=previous_state[1],
                    action=opponent_action,
                    next_state=state[1],
                    reward=reward[1],
                    state_action=previous_state[1] + calculate_one_hot(opponent_action, PK_TEAM_ACTIONS),
                    acc_reward=0
                )
            )

    episode_length = len(episode_opponent_memory)

    episode_player_memory = calc_acc_reward(episode_player_memory, env.winner == 0)
    episode_opponent_memory = calc_acc_reward(episode_opponent_memory, env.winner == 1)

    episode_cut = episode_length - last_n_turns if last_n_turns != None else 0

    return episode_player_memory[episode_cut:] + episode_opponent_memory[episode_cut:]


def calc_acc_reward(memory, is_winner=True):
    acc_reward = 0
    t = len(memory) - 1

    for mem in reversed(memory):
        # reward = mem.reward
        reward = 1 if is_winner and (t == len(memory) - 1) else 0  # only reward 1 for victory, nothing else
        mem.reward = reward
        acc_reward = reward + ((GAMMA ** t) * acc_reward)
        mem.acc_reward = acc_reward
        t -= 1

    return memory


def append_transitions_to_memory(episode_memory, memory):
    for mem in episode_memory:
        memory.append(
            Transition(
                state=torch.as_tensor([mem.state]),  # state
                action=torch.as_tensor([mem.action]),  # action
                next_state=torch.as_tensor([mem.next_state]),  # next state
                reward=torch.as_tensor([mem.reward]),  # reward
                state_action=torch.as_tensor([mem.state_action]),  # state + action
                acc_reward=torch.as_tensor([[mem.acc_reward]])  # acc reward
            )
        )

    return memory


def run_matches(custom_action_fn=None, custom_action_name="Q NN", teams=[], random_teams=True, random_num_of_teams=2):
    if random_teams or len(teams) == 0:
        matches_teams = build_random_teams(random_num_of_teams)
    else:
        matches_teams = teams

    matches_history = []

    random_policy = RandomBattlePolicy()
    greedy_policy = GreedyBattlePolicy()

    def random_policy_get_action(_, state_view): return random_policy.get_action(state_view)
    def greedy_policy_get_action(_, state_view): return greedy_policy.get_action(state_view)

    if len(matches_teams) < 2:
        team0 = matches_teams[0]
        team1 = matches_teams[0]
    else:
        team0, team1 = random.sample(matches_teams, 2)

    def test_batch_wrapper(player_action_fn, opponent_action_fn, team0, team1): return test_batch(
        team0=team0,
        team1=team1,
        player_action_fn=player_action_fn,
        opponent_action_fn=opponent_action_fn
    )

    matches = [
        {
            'player_fn': greedy_policy_get_action,
            'opponent_fn': random_policy_get_action,
            'message': 'Greedy vs random:'
        },
        {
            'player_fn': greedy_policy_get_action,
            'opponent_fn': greedy_policy_get_action,
            'message': 'Greedy vs greedy:'
        },
        {
            'player_fn': custom_action_fn,
            'opponent_fn': random_policy_get_action,
            'message': '{} vs random:'.format(custom_action_name)
        },
        {
            'player_fn': custom_action_fn,
            'opponent_fn': greedy_policy_get_action,
            'message': '{} vs greedy:'.format(custom_action_name)
        },
    ]

    for match in matches:
        winners_1 = test_batch_wrapper(
            player_action_fn=match['player_fn'],
            opponent_action_fn=match['opponent_fn'],
            team0=team0,
            team1=team1
        )
        winners_2 = test_batch_wrapper(
            player_action_fn=match['player_fn'],
            opponent_action_fn=match['opponent_fn'],
            team0=team1,
            team1=team0
        )
        message = '{} {} {}'.format(match['message'], str(winners_1), str(winners_2))
        print(message)
        matches_history.append(message)

    return matches, matches_history
