from framework.DataObjects import get_full_team_view
from framework.behaviour.SelectorPolicies import GUISelectorPolicy
from framework.util.PkmTeamGenerators import RandomGenerator


def main():
    a = GUISelectorPolicy()
    team_gen = RandomGenerator()
    team0 = team_gen.get_team()
    team1 = team_gen.get_team()
    print(a.get_action((get_full_team_view(team0), get_full_team_view(team1))))


if __name__ == '__main__':
    main()
