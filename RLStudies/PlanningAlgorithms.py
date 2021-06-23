import copy

class Pokemon():
    def __init__(self, hp=5, moves=[]) -> None:
        self.hp = hp
        self.moves = moves


class Move():
    def __init__(self, move_name='s', move_power=1, move_accuracy=1) -> None:
        self.move_name = move_name
        self.move_power = move_power
        self.move_accuracy = move_accuracy


class Environment():
    def __init__(self) -> None:
        self.states = []
        self.P = {}


def one_step_lookahead(environment, state, V, discount_factor):
    available_actions = environment.P[state].keys()
    actions = {a: 0 for a in available_actions}
    for action in available_actions:
      transitions = environment.P[state][action]
      for obj in transitions:
        prob = obj[0]
        reward = obj[1]
        next_state = obj[2]
        #here we do not count the probability of taking the action because we want to know what happens if we do take the action
        actions[action] += prob * (reward + (discount_factor * V[next_state]))
    return actions


def value_iteration(environment, discount_factor=0.9, theta=0.0001):
    V = {k: 0 for k in environment.states}
    policy = {k: {} for k in environment.states}
    i = 0

    while True:
        i += 1
        delta = 0
        for state in environment.states:
            v = V[state]
            actions_values = one_step_lookahead(environment, state, V, discount_factor=discount_factor)
            best_action = max(actions_values, key=actions_values.get)
            V[state] = actions_values[best_action]

            available_actions = environment.P[state].keys()
            policy[state] = {a: 0 for a in available_actions}
            policy[state][best_action] = 1

            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            print('Took {} iterations'.format(i))
            break

    return policy, V


def iterative_policy_eval(policy, environment, discount_factor=0.9, theta=0.00001, print_result=False):
    V = {k: 0 for k in environment.states}
    i = 0
    while True:
      i += 1
      delta = 0
      for state in environment.states:
        v = 0
        for move, move_prob in policy[state].items():
          transitions = environment.P[state][move]
          for obj in transitions:
            prob = obj[0]
            reward = obj[1]
            next_state = obj[2]
            v += move_prob * prob * (reward + (discount_factor * V[next_state]))

        delta = max(delta, abs(v - V[state]))
        V[state] = v

      if delta < theta:
        if print_result == True:
          print('Took {} interactions'.format(i))
          print(' ')
          print(V)
        break

    return V


def iterative_policy_improvement(environment, base_policy, discount_factor=0.9):
    policy = copy.deepcopy(base_policy)
    i = 0

    while True:
      i += 1
      V = iterative_policy_eval(policy, environment, discount_factor)

      policy_stable = True

      for state in environment.states:
        #best action by current policy
        chosen_action = max(policy[state], key=policy[state].get)

        #one step look ahead
        actions_values = one_step_lookahead(environment, state, V, discount_factor)
        best_action = max(actions_values, key=actions_values.get)

        if chosen_action != best_action:
          policy_stable = False

        #here we always get the best action and set the probability of choosing it as 1 (super greedy)
        available_actions = environment.P[state].keys()
        policy[state] = {a: 0 for a in available_actions}
        policy[state][best_action] = 1

      if policy_stable == True:
        print('Took {} iterations'.format(i))
        return policy, V


def build_possible_actions(state, moves, opponent):
    actions = {}

    for move in moves:
        actions[move.move_name] = build_possible_transitions(state, move, moves, opponent)

    return actions


def build_possible_transitions(state, move, moves, opponent):
    transitions = []

    if state[0] == '0':
        # terminal state, lost
        transitions.append(
            [1, 0, state]
        )
    elif state[1] == '0':
        # terminal state, won
        transitions.append(
            [1, 0, state]
        )
    else:
        for opp_move in moves:
            multiplier = 0
            if opponent == 'random':
                multiplier = 1 / len(moves)
            elif opponent == 'passive':
                multiplier = 1 if opp_move.move_accuracy == 1 else 0
            elif opponent == 'aggressive':
                multiplier = 1 if opp_move.move_power == 3 else 0

            if multiplier == 0:
                continue

            opp_damage = int(state[0]) - opp_move.move_power
            player_damage = int(state[1]) - move.move_power
            player_hp = 0 if opp_damage <= 0 else opp_damage
            opp_hp = 0 if player_damage <= 0 else player_damage

            player_can_be_killed = player_hp == 0
            opp_can_be_killed = opp_hp == 0

            # player always goes first
            # possible transitions:
            if not player_can_be_killed and not opp_can_be_killed:
                # - player hit, doesn't kill, opp hit, doesn't kill
                transitions.append([multiplier * move.move_accuracy * opp_move.move_accuracy, 0, '{}{}'.format(player_hp, opp_hp)])
                # - player hit, doesn't kill, opp miss, doesn't kill
                transitions.append([multiplier * move.move_accuracy * (1 - opp_move.move_accuracy), 0, '{}{}'.format(state[0], opp_hp)])
                # - player miss, opp hit, doesn't kill
                transitions.append([multiplier * (1 - move.move_accuracy) * opp_move.move_accuracy, 0, '{}{}'.format(player_hp, state[1])])
                # - player miss, opp miss
                transitions.append([multiplier * (1 - move.move_accuracy) * (1 - opp_move.move_accuracy), 0, state])
            elif opp_can_be_killed:
                # - player hit, kill
                transitions.append([move.move_accuracy, 1, '{}0'.format(state[0])])
                # - player miss, opp miss
                transitions.append([multiplier * (1 - move.move_accuracy) * (1 - opp_move.move_accuracy), 0, state])
                if not player_can_be_killed:
                    # - player miss, opp hit, doesn't kill
                    transitions.append([multiplier * (1 - move.move_accuracy) * opp_move.move_accuracy, 0, '{}{}'.format(player_hp, state[1])])
            elif player_can_be_killed:
                # - player hit, doesn't kill, opp hit, kill
                transitions.append([multiplier * move.move_accuracy * opp_move.move_accuracy, -1, '0{}'.format(opp_hp)])
                # - player miss, opp hit, kill
                transitions.append([multiplier * (1 - move.move_accuracy) * opp_move.move_accuracy, -1, '0{}'.format(state[1])])

    unique_transitions = [list(x) for x in set(tuple(x) for x in transitions)]

    return unique_transitions


def build_random_equiprob_policy(states, moves):
    policy = {}

    for state in states:
        policy[state] = {}
        for move in moves:
            policy[state][move.move_name] = 1 / len(moves)

    return policy


if __name__ == "__main__":
    strong_move = Move('s', 3, 0.5)
    medium_move = Move('m', 2, 0.75)
    weak_move = Move('w', 1, 1)
    moves = [strong_move, medium_move, weak_move]

    pokemon_1 = Pokemon(hp=5, moves=[strong_move, medium_move, weak_move])
    pokemon_2 = Pokemon(hp=5, moves=[strong_move, medium_move, weak_move])

    opponents = [
        'aggressive',  # always use strongest move
        'passive',  # always use accuraciest (?) move
        'random'  # choose randomly
    ]
    environment = Environment()
    environment.states = ['{}{}'.format(i, j) for i in range(6) for j in range(6)]

    random_policy = build_random_equiprob_policy(environment.states, moves)

    for opponent in opponents:
        environment.P = {state: build_possible_actions(state, moves, opponent) for state in environment.states}
        print('Opponent: {}'.format(opponent))
        print('Policy iteration')
        policy, V = iterative_policy_improvement(environment, random_policy)
        print(V)
        print(policy)
        print('Value iteration')
        policy, V = value_iteration(environment)
        print(V)
        print(policy)
