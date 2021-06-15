import torch
from framework.behaviour.BattlePolicies import estimate_damage
from framework.DataTypes import PkmStat
from framework.DataConstants import DEFAULT_PARTY_SIZE, DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS, MOVE_POWER_MAX, TYPE_CHART_MULTIPLIER

torch.manual_seed(0)

PK_TEAM_ACTIONS = DEFAULT_PKM_N_MOVES + DEFAULT_PARTY_SIZE


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

    std_move_choices = [move_choice / MOVE_POWER_MAX for move_choice in move_choices]
    std_switch_choices = [switch_choice / 2 for switch_choice in switch_choices]

    return [my_active.hp / MAX_HIT_POINTS, opp_active_hp / MAX_HIT_POINTS] + std_move_choices + std_switch_choices


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
