from competitor.greedy.Greedy import Greedy
from framework.competition.CompetitionObjects import Match, ExampleCompetitor, GUIExampleCompetitor
from framework.util.PkmTeamGenerators import RandomGenerator

def main():
    rg = RandomGenerator()
    c0 = Greedy(team=rg.get_team(), name="Player0")
    c1 = ExampleCompetitor(rg.get_team(), name="Player1")
    m = Match(c0, c1, debug=True)
    m.run()


if __name__ == '__main__':
    main()
