
import torch

from framework.DataObjects import MetaData, PkmFullTeam, GameStateView

from framework.behaviour import BattlePolicy
from framework.behaviour.DataAggregators import NullDataAggregator
from framework.competition.CompetitionObjects import Competitor

from RLStudies.QValueNN import QValueNN
from RLStudies.Utils import build_state_representation, select_player_action_from_q_value


class QValueBattlePolicy(BattlePolicy):

    def __init__(self, file_name='') -> None:
        super().__init__()
        self.q_value_nn = QValueNN(input_size=16)
        self.q_value_nn.load_state_dict(torch.load(file_name))
        self.q_value_nn.eval()

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameStateView) -> int:
        return select_player_action_from_q_value(build_state_representation(g), self.q_value_nn)

class QValueCompetitor(Competitor):

    def __init__(self, name: str = "QValueCompetitor", team: PkmFullTeam = None):
        self._name = name
        self._battle_policy = QValueBattlePolicy()
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
