from competitor.greedy.Greedy import Greedy
from competitor.random.Random import Random
from framework.competition.CompetitionObjects import Match, GUIExampleCompetitor
from framework.util.PkmTeamGenerators import RandomGenerator

def main():
    rg = RandomGenerator()
    c0 = Greedy(team=rg.get_team())
    c1 = Greedy(team=rg.get_team())
    m = Match(c0, c1, debug=True, n_games=10)
    m.run()


if __name__ == '__main__':
    main()
