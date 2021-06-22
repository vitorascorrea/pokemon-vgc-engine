from RLStudies.Tester import build_random_teams
from RLStudies.QValueCompetitor import QValueBattlePolicy
from framework.behaviour.BattlePolicies import GUIBattlePolicy, RandomBattlePolicy
from framework.process.BattleEngine import PkmBattleEnv


def main():
    file_name = './RLStudies/saves/reward_1_8_teams_simple.pth'
    team0, team1 = build_random_teams(2)
    t = False
    a0 = GUIBattlePolicy()
    a1 = QValueBattlePolicy(file_name=file_name)
    ep = 0
    n_battles = 1

    while ep < n_battles:
        env = PkmBattleEnv(debug=True, teams=[team0, team1])
        s = env.reset()
        v = env.game_state_view
        env.render()
        ep += 1
        while not t:
            o0 = s[0] if a0.requires_encode() else v[0]
            o1 = s[1] if a1.requires_encode() else v[1]
            a = [a0.get_action(o0), a1.get_action(o1)]
            s, _, t, v = env.step(a)
            env.render()
        t = False
    a0.close()


if __name__ == '__main__':
    main()
