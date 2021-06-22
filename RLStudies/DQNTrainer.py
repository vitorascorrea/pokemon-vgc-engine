import math
import random
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from framework.behaviour.BattlePolicies import RandomBattlePolicy
from competitor.greedy.Greedy import GreedyBattlePolicy
from framework.DataConstants import DEFAULT_N_ACTIONS, DEFAULT_PARTY_SIZE, DEFAULT_TEAM_SIZE, MAX_HIT_POINTS
from RLStudies.Tester import build_random_teams
from RLStudies.Utils import Transition, append_transitions_to_memory, build_state_representation, run_matches, run_episode

from framework.process.BattleEngine import PkmBattleEnv

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 100000
SAVE_EPISODE = 10000
TEST_EPISODE = 500
MEMORY = 10000
BATCH_SIZE = 32
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20000
TARGET_UPDATE = 1
GREEDY_POLICY = GreedyBattlePolicy()
RANDOM_POLICY = RandomBattlePolicy()

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    # inspired from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, input_size=9, output_size=DEFAULT_N_ACTIONS, target=False):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.nn = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_size),
        )

        if target:
            self.freeze_parameters()

    def forward(self, x):
        x = self.flatten(x)
        return self.nn(x)

    def freeze_parameters(self):
        for p in self.nn.parameters():
            p.requires_grad = False


class DQNTrainer():
    def __init__(self, policy_net=None, e_greedy_decay=None, double_dqn=False, random_teams=True, fixed_teams=None) -> None:
        self.policy_net = policy_net if policy_net else Net()
        self.target_net = Net()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.freeze_parameters()
        self.target_net.eval()

        self.double_dqn = double_dqn

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.loss_fn = F.smooth_l1_loss
        self.memory = ReplayMemory(MEMORY)

        self.steps_counter = 0
        self.current_loss = 0
        self.total_loss = 0
        self.last_reward_value = 0

        self.random_teams = random_teams
        self.fixed_teams = fixed_teams

        self.e_greedy_decay = e_greedy_decay

    def select_action(self, state, episode=0, use_nn_value=False):
        sample = random.random()
        eps_threshold = 0
        if episode != None:
            eps_threshold = self.e_greedy_decay(episode)

        if sample > eps_threshold or use_nn_value:
            if use_nn_value:
                self.policy_net.eval()

            with torch.no_grad():
                action_values = self.policy_net(torch.as_tensor([state]))
                max_action = torch.argmax(action_values)
                return max_action.item()
        else:
            return random.randrange(DEFAULT_N_ACTIONS)

    def select_action_wrapper(self, episode=0, use_nn_value=True):
        return lambda _, state_view: self.select_action(build_state_representation(state_view), episode=episode, use_nn_value=use_nn_value)

    def select_opponent_wrapper(self):
        threshold = random.random()

        if threshold > 0.5:
            return lambda _, state_view: GREEDY_POLICY.get_action(state_view)
        else:
            return lambda _, state_view: RANDOM_POLICY.get_action(state_view)

    def train(self, file_name="DQN"):
        for i_episode in range(NUM_EPISODES):
            team0, team1 = self.get_teams()
            env = PkmBattleEnv(debug=True, teams=[team0, team1])

            self.policy_net.train()

            # reset last reward value for the episode
            self.last_reward_value = 0

            episode_memory = run_episode(
                env,
                store_opponent_memory=True,
                player_action_fn=self.select_action_wrapper(episode=i_episode, use_nn_value=False),
                opponent_action_fn=self.select_opponent_wrapper(),
                compute_reward_fn=self.compute_reward,
                step_fn=lambda t: self.optimize_model_and_update_target(t)
            )

            self.memory = append_transitions_to_memory(episode_memory, self.memory)

            if i_episode > 0 and i_episode % TEST_EPISODE == 0:
                print("Episode {} Current loss: {}".format(i_episode, self.current_loss))
                run_matches(custom_action_fn=self.select_action_wrapper(use_nn_value=True), custom_action_name="DQN", random_teams=self.random_teams, teams=[team0, team1])

        torch.save(self.policy_net.state_dict(), './competitor/DQN/saves/{}_final.pth'.format(file_name))

    def get_teams(self):
        if self.random_teams or not self.fixed_teams:
            return build_random_teams()
        else:
            fixed_teams_length = len(self.fixed_teams) - 1
            team_0 = None
            team_1 = None

            while team_0 == team_1:
                team_0 = random.randint(0, fixed_teams_length)
                team_1 = random.randint(0, fixed_teams_length)

            return self.fixed_teams[team_0], self.fixed_teams[team_1]

    def optimize_model_and_update_target(self, t):
        if len(self.memory) > BATCH_SIZE:
            # Perform one step of the optimization (on the policy network)
            loss = self.optimize_model(t)

            self.steps_counter += 1
            self.total_loss += loss
            self.current_loss = self.total_loss / self.steps_counter

            if self.steps_counter % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.freeze_parameters()
                self.target_net.eval()

    def optimize_model(self, t):
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        expected_q = None
        policy_net_pred = self.policy_net(state_batch)
        current_q_values = policy_net_pred.gather(1, action_batch.unsqueeze(1))
        target_net_pred = self.target_net(next_state_batch).detach()

        if self.double_dqn:
            next_state_values = self.policy_net(next_state_batch).detach()
            _, a_prime = next_state_values.max(1)
            q_target_s_a_prime = target_net_pred.gather(1, a_prime.unsqueeze(1))
            q_target_s_a_prime = q_target_s_a_prime.squeeze()

            # Compute the expected Q values
            expected_q = (q_target_s_a_prime * GAMMA) + reward_batch
        else:
            max_next_q, _ = target_net_pred.max(1)
            # Compute the expected Q values
            expected_q = (max_next_q * GAMMA) + reward_batch

        # Compute loss
        loss = self.loss_fn(current_q_values, expected_q.unsqueeze(1).detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_reward(self, state_view=None, winner=False):
        hp_value = 1
        fainted_value = 2
        victory_value = 30

        current_value = 0

        my_team = state_view.get_team_view(0)
        my_active = my_team.active_pkm_view
        my_party = [my_team.get_party_pkm_view(i) for i in range(DEFAULT_PARTY_SIZE)]
        my_team_pkms = [my_active] + my_party

        for pkm in my_team_pkms:
            current_value += (pkm.hp / MAX_HIT_POINTS) * hp_value
            if pkm.hp == 0:
                current_value -= fainted_value

        current_value += (DEFAULT_TEAM_SIZE - len(my_team_pkms)) * hp_value

        opp_team = state_view.get_team_view(1)
        opp_active = opp_team.active_pkm_view
        opp_party = [opp_team.get_party_pkm_view(i) for i in range(DEFAULT_PARTY_SIZE)]
        opp_team_pkmns = [opp_active] + opp_party

        for pkm in opp_team_pkmns:
            current_value -= (pkm.hp / MAX_HIT_POINTS) * hp_value
            if pkm.hp == 0:
                current_value += fainted_value

        current_value -= (DEFAULT_TEAM_SIZE - len(opp_team_pkmns)) * hp_value

        if winner == True:
            current_value += victory_value
        elif winner == False:
            current_value -= victory_value

        to_return = current_value - self.last_reward_value
        self.last_reward_value = current_value

        return to_return


class EGreedyDecay():
    @staticmethod
    def epsilon_linear_decay(episode):
        return EPS_END + (EPS_START - EPS_END) * (1 - (episode / NUM_EPISODES))

    @staticmethod
    def epsilon_exponential_decay(episode):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-1 * episode / EPS_DECAY)


if __name__ == "__main__":
    policy_net = Net()
    teams = build_random_teams(2)
    with open('./competitor/DQN/teams/random_fixed_teams.pickle', 'wb') as handle:
        pickle.dump(teams, handle)

    dqn_agent = DQNTrainer(
        policy_net=policy_net,
        double_dqn=True,
        e_greedy_decay=EGreedyDecay.epsilon_linear_decay,
        random_teams=False,
        fixed_teams=teams
    )
    dqn_agent.train()
    run_matches(custom_action_fn=dqn_agent.select_action_wrapper(), custom_action_name="DQN", random_teams=False, teams=teams)
