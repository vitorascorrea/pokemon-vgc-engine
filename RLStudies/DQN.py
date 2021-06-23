import numpy as np

from typing import List

from framework.DataObjects import MetaData, PkmFullTeam, GameStateView
from framework.DataConstants import DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_PKM_N_MOVES
from framework.DataTypes import PkmStat

from framework.behaviour import BattlePolicy
from framework.behaviour.BattlePolicies import estimate_damage
from framework.behaviour.DataAggregators import NullDataAggregator
from framework.competition.CompetitionObjects import Competitor


class DQNBattlePolicy(BattlePolicy):

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameStateView) -> int:
        # check weather condition
        weather = g.weather_condition

        # get my team
        my_team = g.get_team_view(0)
        my_active = my_team.active_pkm_view
        my_active_type = my_active.type
        my_party = [my_team.get_party_pkm_view(i) for i in range(DEFAULT_PARTY_SIZE)]
        my_active_moves = [my_active.get_move_view(i) for i in range(DEFAULT_PKM_N_MOVES)]
        my_attack_stage = my_team.get_stage(PkmStat.ATTACK)

        # get opp team
        opp_team = g.get_team_view(1)
        opp_active = opp_team.active_pkm_view
        opp_active_type = opp_active.type
        opp_active_hp = opp_active.hp
        opp_defense_stage = opp_team.get_stage(PkmStat.DEFENSE)

        # get best move
        damage: List[float] = []
        for move in my_active_moves:
            damage.append(estimate_damage(move.type, my_active_type, move.power, opp_active_type, my_attack_stage, opp_defense_stage, weather))
        move_id = int(np.argmax(damage))

        # switch decision
        best_pkm = 0
        if opp_active_hp > damage[move_id]:
            effectiveness_to_stay = TYPE_CHART_MULTIPLIER[my_active_type][opp_active_type]
            for i, pkm in enumerate(my_party):
                effectiveness_party = TYPE_CHART_MULTIPLIER[pkm.type][opp_active_type]
                if effectiveness_party > effectiveness_to_stay and pkm.hp != 0.0:
                    effectiveness_to_stay = effectiveness_party
                    best_pkm = i
        if best_pkm > 0:
            move_id = DEFAULT_PKM_N_MOVES + best_pkm

        return move_id


class DQN(Competitor):

    def __init__(self, name: str = "DQN", team: PkmFullTeam = None):
        self._name = name
        self._battle_policy = DQNBattlePolicy()
        self._team = team

    @property
    def name(self):
        return self._name

    def reset(self):
        pass

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy

    @property
    def meta_data(self) -> MetaData:
        return NullDataAggregator.null_metadata

    def want_to_change_team(self):
        return True

    @property
    def team(self) -> PkmFullTeam:
        return self._team

    @team.setter
    def team(self, team: PkmFullTeam):
        self._team = team
