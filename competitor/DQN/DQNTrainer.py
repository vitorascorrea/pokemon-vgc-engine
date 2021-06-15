import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from framework.process.BattleEngine import PkmBattleEnv
from framework.util.PkmTeamGenerators import RandomGenerator
from competitor.greedy.Greedy import GreedyBattlePolicy
from competitor.random.Random import RandomBattlePolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 100000
SAVE_EPISODE = 10000
TEST_EPISODE = 1000
MEMORY = 1000000
BATCH_SIZE = 1000
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20000
TARGET_UPDATE = 10
RANDOM_TEAM_UPDATE = 5001

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # inspired from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, n_actions, target=False):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1383, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.ReLU()
        )

        if target:
            for p in self.linear_relu_stack.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def optimize_model(optimizer, memory, policy_net, target_net):
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def select_action(state, policy_net, n_actions, episode=None):
    sample = random.random()
    eps_threshold = 0
    if episode != None:
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * (1 - (episode / NUM_EPISODES))
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * episode / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():
            action_values = policy_net(state)
            return torch.argmax(action_values)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=DEVICE, dtype=torch.long)


def train(team0=None, team1=None, opponent_policy=GreedyBattlePolicy(), self_training=False, store_opponent_memory=False, file_name='nn', start_from_episode=0, random_teams=False):
    if random_teams or not team0 or not team1:
        team0, team1 = build_random_teams()

    env = PkmBattleEnv(debug=True, teams=[team0, team1])
    n_actions = env.action_space.n

    policy_net = DQN(n_actions=n_actions).to(DEVICE)

    if start_from_episode > 0:
        policy_net.load_state_dict(torch.load('./competitor/DQN/saves/{}_{}.pth'.format(file_name, start_from_episode)))

    target_net = DQN(n_actions=n_actions, target=True).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
    memory = ReplayMemory(MEMORY)
    current_loss = 0
    total_loss = 0
    team = 0

    for i_episode in range(start_from_episode, NUM_EPISODES):
        if i_episode > start_from_episode and i_episode % SAVE_EPISODE == 0:
            torch.save(policy_net.state_dict(), './competitor/DQN/saves/{}_{}.pth'.format(file_name, i_episode))

        if i_episode > start_from_episode and i_episode % TEST_EPISODE == 0:
            winners = test_batch(opponent_policy=opponent_policy, player_nn=policy_net)
            print('Episode: {} Team: {} Loss: {}'.format(i_episode, team, current_loss))
            print(winners)
            policy_net.train()

        if i_episode > 0 and i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

        if (random_teams or not team0 or not team1) and i_episode % RANDOM_TEAM_UPDATE == 0:
            team0, team1 = build_random_teams()
            env = PkmBattleEnv(debug=True, teams=[team0, team1])
            team += 1

        previous_state = None
        state = env.reset()
        state_view = env.game_state_view

        done = False
        turns = 0

        while not done:
            turns += 1
            if turns > 1000:
                break

            player_action = select_action(
                state=torch.as_tensor([state[0]]),
                policy_net=policy_net,
                n_actions=n_actions,
                episode=i_episode
            ).item()

            opponent_action = None

            if self_training:
                opponent_action = select_action(
                    state=torch.as_tensor([state[1]]),
                    policy_net=policy_net,
                    n_actions=n_actions,
                    episode=i_episode
                ).item()
            else:
                opponent_action = opponent_policy.get_action(state_view[1])

            action = [player_action, opponent_action]

            previous_state = state

            state, reward, done, state_view = env.step(action)

            # Store the transition in memory
            memory.push(
                torch.as_tensor([previous_state[0]]),
                torch.as_tensor([[player_action]]),
                torch.as_tensor([state[0]]),
                torch.as_tensor([reward[0]])
            )

            if store_opponent_memory:
                memory.push(
                    torch.as_tensor([previous_state[1]]),
                    torch.as_tensor([[opponent_action]]),
                    torch.as_tensor([state[1]]),
                    torch.as_tensor([reward[1]])
                )

            # Perform one step of the optimization (on the policy network)
            if len(memory) > BATCH_SIZE:
                total_loss += optimize_model(optimizer=optimizer, memory=memory, policy_net=policy_net, target_net=target_net)
                current_loss = total_loss / i_episode

    torch.save(policy_net.state_dict(), './competitor/DQN/saves/{}_final.pth'.format(file_name))
    return policy_net, target_net


def test(team0=None, team1=None, opponent_policy=GreedyBattlePolicy(), random_teams=False, file_name='nn_final', player_nn=None):
    if random_teams or not team0 or not team1:
        team0, team1 = build_random_teams()

    env = PkmBattleEnv(debug=True, teams=[team0, team1])
    n_actions = env.action_space.n

    state = env.reset()
    state_view = env.game_state_view

    policy_net = player_nn

    if not player_nn:
        policy_net = DQN(n_actions=n_actions).to(DEVICE)
        policy_net.load_state_dict(torch.load('./competitor/DQN/saves/{}.pth'.format(file_name)))

    policy_net.eval()

    # env.render()
    done = False
    turn = 0
    while not done:
        turn += 1
        if turn > 1000:
            break
        player_action = select_action(
            state=torch.as_tensor([state[0]]),
            policy_net=policy_net,
            n_actions=n_actions
        ).item()
        opponent_action = opponent_policy.get_action(state_view[1])
        action = [player_action, opponent_action]
        state, reward, done, state_view = env.step(action)
        # env.render()

    return env.winner


def test_batch(num_tests=100, team0=None, team1=None, random_teams=False, opponent_policy=GreedyBattlePolicy(), file_name='nn_final', player_nn=None):
    winners = {
        -1: 0,
        0: 0,
        1: 0
    }

    for i in range(num_tests):
        winner = test(team0=team0, team1=team1, random_teams=random_teams, opponent_policy=opponent_policy, file_name=file_name, player_nn=player_nn)
        winners[winner] += 1

    return winners


def build_random_teams():
    rg = RandomGenerator()
    team0 = rg.get_team().get_battle_team(random.sample(range(6), 4))
    team1 = rg.get_team().get_battle_team(random.sample(range(6), 4))

    return team0, team1


if __name__ == "__main__":
    random_policy = RandomBattlePolicy()
    greedy_policy = GreedyBattlePolicy()
    team0, team1 = build_random_teams()

    policy_net, target_net = train(opponent_policy=greedy_policy, random_teams=True, file_name='greedy_supervised_bla', store_opponent_memory=True)
    winners = test_batch(opponent_policy=greedy_policy, player_nn=policy_net, num_tests=1000)
    print(winners)
