from framework.DataObjects import MetaData, PkmFullTeam, TeamValue
from framework.behaviour import TeamValuator


class NullTeamValue(TeamValue):

    def compare_to(self, value) -> int:
        return 0


class TeamValuation:
    null_team_value = NullTeamValue()

    def __init__(self, tv: TeamValuator, pkm_full_team: PkmFullTeam, meta_data: MetaData):
        self.__tv = tv
        self.__pkm_full_team = pkm_full_team
        self.__meta_data = meta_data
        # output
        self.__value = TeamValuation.null_team_value

    # noinspection PyBroadException
    def run(self):
        try:
            self.__value = self.__tv.get_action((self.__pkm_full_team, self.__meta_data))
        except:
            self.__value = TeamValuation.null_team_value

    @property
    def team_value(self) -> TeamValue:
        return self.__value
