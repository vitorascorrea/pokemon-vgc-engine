import torch
import torch.nn as nn
import torch.optim as optim

from framework.process.BattleEngine import PkmBattleEnv
from competitor.greedy.Greedy import GreedyBattlePolicy
from framework.DataConstants import DEFAULT_PARTY_SIZE, DEFAULT_PKM_N_MOVES
from RLStudies.Tester import build_random_teams
from RLStudies.Utils import TransitionIterator, append_transitions_to_memory, build_state_representation, run_episode, run_matches, select_player_action_from_q_value
from RLStudies.DQNTrainer import Net

torch.manual_seed(0)

EPISLON = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PK_TEAM_ACTIONS = DEFAULT_PKM_N_MOVES + DEFAULT_PARTY_SIZE


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
    greedy_policy_get_action = lambda _, state_view: greedy_policy.get_action(state_view, EPISLON)

    for _ in range(num_episodes):
        if random_teams:
            teams = build_random_teams(random_num_of_teams)

        for team0 in teams:
            for team1 in teams:
                env = PkmBattleEnv(debug=True, teams=[team0, team1])
                episode_memory = run_episode(env=env, player_action_fn=greedy_policy_get_action, opponent_action_fn=greedy_policy_get_action, last_n_turns=last_n_turns)
                memory = append_transitions_to_memory(episode_memory, memory)


    batch = TransitionIterator(*zip(*memory))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_batch = torch.cat(batch.state_action)
    acc_reward_batch = torch.cat(batch.acc_reward)

    return state_batch, action_batch, reward_batch, state_action_batch, acc_reward_batch


def q_value_get_action_wrapper(q_value_nn):
    return lambda _, state_view: select_player_action_from_q_value(build_state_representation(state_view), q_value_nn=q_value_nn)


if __name__ == "__main__":
    q_value_nn = Net(input_size=16, output_size=1)
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
            run_matches(custom_action_fn=q_value_get_action_wrapper(q_value_nn=q_value_nn), custom_action_name="Q NN", teams=teams, random_teams=random_teams, random_num_of_teams=num_of_teams)

        if early_stop and early_stopper.step(epoch, current_loss):
            print('Early stopping!')
            break

    torch.save(q_value_nn.state_dict(), file_name)
