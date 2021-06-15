
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple
from framework.process.BattleEngine import PkmBattleEnv
from competitor.greedy.Greedy import GreedyBattlePolicy
from competitor.random.Random import RandomBattlePolicy
from framework.DataConstants import DEFAULT_PARTY_SIZE, DEFAULT_PKM_N_MOVES
from competitor.DQN.Tester import test_batch, build_random_teams
from competitor.DQN.QValueHelpers import build_state_representation, calculate_one_hot, select_player_action_from_q_value

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.999
EPISLON = 0.1
PK_TEAM_ACTIONS = DEFAULT_PKM_N_MOVES + DEFAULT_PARTY_SIZE

TransitionIterator = namedtuple('TransitionIterator',
                                ('state', 'action', 'next_state', 'reward', 'state_action',  'acc_reward'))

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


class QValueNN(nn.Module):

    def __init__(self, input_size=0):
        super(QValueNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 15),
            nn.ELU(),
            nn.Linear(15, 1),
            nn.ELU(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class EarlyStopping():
    def __init__(self, min_delta=0, patience=50, warm_up_epochs=0, percentage=False, verbose=False) -> None:
        self.min_delta = min_delta
        self.patience = patience
        self.percentage = percentage
        self.warm_up_epochs = warm_up_epochs
        self.best = None
        self.num_of_bad_epochs = 0
        self.verbose = verbose

    def step(self, epoch, current_loss):
        if epoch < self.warm_up_epochs:
            return False

        if self.best == None:
            self.best = current_loss
            return False

        if self.is_better(current_loss):
            self.best = current_loss
            self.num_of_bad_epochs = 0
            return False
        else:
            self.num_of_bad_epochs += 1
            if self.verbose:
                print('Epochs without improvement: {} Current: {} Best: {}'.format(self.num_of_bad_epochs, current_loss, self.best))

        return self.num_of_bad_epochs > self.patience

    def is_better(self, current):
        if not self.percentage:
            return current < self.best + self.min_delta
        else:
            return current < self.best + (self.best * self.min_delta)


def generate_history(teams=[], random_teams=False, random_num_of_teams=2, num_episodes=1000, last_n_turns=None):
    memory = []
    greedy_policy = GreedyBattlePolicy()

    for _ in range(num_episodes):
        if random_teams:
            teams = build_random_teams(random_num_of_teams)

        for team0 in teams:
            for team1 in teams:
                env = PkmBattleEnv(debug=True, teams=[team0, team1])
                episode_memory = run_episode(env=env, player_policy=greedy_policy, opponent_policy=greedy_policy, last_n_turns=last_n_turns)
                memory = append_transitions_to_memory(episode_memory, memory)


    batch = TransitionIterator(*zip(*memory))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_batch = torch.cat(batch.state_action)
    acc_reward_batch = torch.cat(batch.acc_reward)

    return state_batch, action_batch, reward_batch, state_action_batch, acc_reward_batch


def run_episode(env, player_policy, opponent_policy, last_n_turns=None):
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

        player_action = player_policy.get_action(state_view[0], EPISLON)
        opponent_action = opponent_policy.get_action(state_view[1], EPISLON)

        action = [player_action, opponent_action]

        previous_state = state
        previous_state_view = state_view

        state, reward, done, state_view = env.step(action)

        previous_state[0] = build_state_representation(previous_state_view[0])
        previous_state[1] = build_state_representation(previous_state_view[1])

        state[0] = build_state_representation(state_view[0])
        state[1] = build_state_representation(state_view[1])

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


def calc_acc_reward(memory, is_winner=True):
    acc_reward = 0
    t = len(memory) - 1

    for mem in reversed(memory):
        # reward = mem.reward
        reward = 1 if is_winner and (t == len(memory) - 1) else 0 # only reward 1 for victory, nothing else
        acc_reward = reward + ((GAMMA ** t) * acc_reward)
        mem.acc_reward = acc_reward
        t -= 1

    return memory


def run_matches(q_value_nn=None, teams=[], random_teams=True, random_num_of_teams=2):
    if random_teams or len(teams) == 0:
        matches_teams = build_random_teams(random_num_of_teams)
    else:
        matches_teams = teams

    random_policy = RandomBattlePolicy()
    greedy_policy = GreedyBattlePolicy()

    random_policy_get_action = lambda _, state_view: random_policy.get_action(state_view)
    greedy_policy_get_action = lambda _, state_view: greedy_policy.get_action(state_view)
    q_value_get_action = lambda _, state_view: select_player_action_from_q_value(build_state_representation(state_view), q_value_nn=q_value_nn)

    if len(matches_teams) < 2:
        team0 = matches_teams[0]
        team1 = matches_teams[0]
    else:
        team0, team1 = random.sample(matches_teams, 2)

    test_batch_wrapper = lambda player_action_fn, opponent_action_fn, team0, team1: test_batch(
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
            'player_fn': q_value_get_action,
            'opponent_fn': random_policy_get_action,
            'message': 'Q NN vs random:'
        },
        {
            'player_fn': q_value_get_action,
            'opponent_fn': greedy_policy_get_action,
            'message': 'Q NN vs greedy:'
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
        print('{} {} {}'.format(match['message'], str(winners_1), str(winners_2)))


if __name__ == "__main__":
    q_value_nn = QValueNN(input_size=16)
    file_name = './competitor/DQN/saves/reward_1_random_teams_simple.pth'

    optimizer = optim.Adam(q_value_nn.parameters())
    loss_fn = nn.SmoothL1Loss()

    num_of_teams = 8
    random_teams = True
    last_n_turns = None
    num_episodes = 2000
    num_epochs = 10000
    batch_size = None

    early_stop = False

    early_stopper = EarlyStopping(
        min_delta=0.05,
        patience=100,
        warm_up_epochs=1500,
        percentage=True,
        verbose=True
    )

    teams = build_random_teams(num_of_teams)
    state_batch, action_batch, reward_batch, state_action_batch, acc_reward_batch = generate_history(
        teams=teams,
        random_teams=random_teams,
        random_num_of_teams=num_of_teams,
        num_episodes=num_episodes,
        last_n_turns=last_n_turns
    )

    cumulative_loss = 0
    epochs_no_improve = 0
    min_loss = float('inf')

    for epoch in range(num_epochs + 1):
        q_value_nn.train()
        optimizer.zero_grad()

        idx = torch.randperm(state_action_batch.shape[0])[:batch_size]
        random_state_action_batch = state_action_batch[idx]
        random_predicted_acc_rewards_batch = acc_reward_batch[idx]

        predicted_acc_rewards = q_value_nn(random_state_action_batch)
        loss = loss_fn(predicted_acc_rewards, random_predicted_acc_rewards_batch)
        current_loss = loss.item()
        cumulative_loss += current_loss

        # Optimize the model
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 and epoch > 0:
            print('epoch {}: {}'.format(epoch, cumulative_loss / epoch))
            run_matches(q_value_nn=q_value_nn, teams=teams, random_teams=random_teams, random_num_of_teams=num_of_teams)

        if early_stop and early_stopper.step(epoch, current_loss):
            print('Early stopping!')
            break

    torch.save(q_value_nn.state_dict(), file_name)