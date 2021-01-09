from Engine.BattleEngine import PkmBattleEngine
from Engine.PkmTeamGenerator import RandomGenerator
from Behaviour.BattlePolicies import HeuristicBattlePolicy
from Behaviour.BattlePolicies import GUIBattlePolicy
from Behaviour.RandomBattleAgent import RandomBattleAgent
from Util.Recorder import Recorder


def main():
    env = PkmBattleEngine(debug=True)
    env.set_team_generator(RandomGenerator())
    env.reset()  # set correct team size for get_n_party
    t = False
    a0 = GUIBattlePolicy(env.trainer_view[0].get_n_party())
    a1 = RandomBattleAgent()
    r = Recorder(name="random_agent")
    ep = 0
    n_battles = 3
    while ep < n_battles:
        s = env.reset()
        v = env.trainer_view
        env.render()
        ep += 1
        while not t:
            o0 = s[0] if a0.requires_encode() else v[0]
            o1 = s[1] if a1.requires_encode() else v[1]
            a = [a0.get_action(o0), a1.get_action(o1)]
            r.record((s[0], a[0], ep))
            s, _, t, v = env.step(a)
            env.render()
        t = False
    r.save()
    a0.close()


if __name__ == '__main__':
    main()