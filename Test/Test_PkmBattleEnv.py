from Engine.PkmBattleEngine import PkmBattleEngine
from Engine.PkmTeamGenerator import RandomGenerator
from Player.HeuristicAgent import HeuristicAgent
from Player.GUIAgent import GUIAgent
from Player.RandomAgent import RandomAgent
from Util.Recorder import Recorder


def main():
    env = PkmBattleEngine(debug=True)
    env.set_team_generator(RandomGenerator())
    t = False
    a0 = RandomAgent()
    a1 = RandomAgent()
    r = Recorder(name="random_agent")
    ep = 0
    n_matches = 1000
    while ep < n_matches:
        s = env.reset()
        v = env.trainer_view
        # env.render()
        ep += 1
        while not t:
            a = [a0.get_action(v[0]), a1.get_action(v[1])]
            r.record((s[0], a[0], ep))
            s, _, t, v = env.step(a)
            # env.render()
        t = False
    r.save()
    a0.close()


if __name__ == '__main__':
    main()