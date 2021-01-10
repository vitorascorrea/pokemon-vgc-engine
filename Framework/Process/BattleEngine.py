import gym
import random
import numpy as np

from gym import spaces
from typing import List, Tuple

from Behaviour import BattlePolicy
from Framework.Competition.Config import N_BATTLES
from Framework.Competition.PkmTeamGenerator import PkmTeamGenerator
from Framework.DataConstants import N_MOVES, N_SWITCHES, MAX_HIT_POINTS, STATE_DAMAGE, SPIKES_2, SPIKES_3, MOVE_MAX_PP, \
    TYPE_CHART_MULTIPLIER
from Framework.DataObjects import PkmTeam, Pkm
from Framework.DataTypes import WeatherCondition, PkmEntryHazard, PkmType, PkmStatus, PkmStat, N_TYPES, N_STATUS, \
    N_STATS, N_STAGES, N_ENTRY_HAZARD, N_HAZARD_STAGES, N_WEATHER, MIN_STAGE, MAX_STAGE
from Framework.StandardPkmMoves import Struggle
from Util import Recorder
from Util.Encoding import one_hot


class PkmBattleEnv(gym.Env):
    def __init__(self, teams: List[PkmTeam] = None, debug: bool = False):

        # random active pokemon
        self.team_view = None
        if teams is None:
            self.teams: List[PkmTeam] = [PkmTeam(), PkmTeam()]
        else:
            self.teams: List[PkmTeam] = teams
        self.weather: WeatherCondition = WeatherCondition.CLEAR
        self.n_turns_no_clear: int = 0
        self.switched = [False, False]
        self.turn = 0
        self.move_view = self.__create_pkm_move_view()
        self.trainer_view = [self.__create_trainer_view(0), self.__create_trainer_view(1)]
        self.debug = debug
        self.log = ''
        self.action_space = spaces.Discrete(N_MOVES + N_SWITCHES)
        self.observation_space = spaces.Discrete(len(self.trainer_view[0].encode()))
        self.team_generator = None
        self.winner = -1

    def set_team_generator(self, team_generator: PkmTeamGenerator):
        self.team_generator = team_generator

    def step(self, actions):

        # Reset variables
        r = [0., 0.]
        t = [False, False]
        if self.debug:
            self.turn += 1
            self.log = 'TURN %s\n\n' % str(self.turn)

        # switch pkm
        self.__process_switch_pkms(actions)

        # set trainer attack order
        first, second = self.__get_attack_order(actions)
        first_team = self.teams[first]
        first_pkm = first_team.active
        second_team = self.teams[second]
        second_pkm = second_team.active

        # get entry hazard damage
        dmg_2_first = self.__get_entry_hazard_damage(first)
        dmg_2_second = self.__get_entry_hazard_damage(second)

        r[first] = (dmg_2_second - dmg_2_first) / MAX_HIT_POINTS
        r[second] = (dmg_2_first - dmg_2_second) / MAX_HIT_POINTS

        active_not_fainted = not (first_pkm.fainted() or second_pkm.fainted())

        # process all pre battle effects
        self.__process_pre_battle_effects()

        # confusion state damage
        dmg_2_first = self.__get_pre_combat_damage(first) if active_not_fainted else 0.
        dmg_2_second = self.__get_pre_combat_damage(second) if active_not_fainted else 0.

        first_confusion_damage = dmg_2_first > 0.
        second_confusion_damage = dmg_2_second > 0.

        r[first] += (dmg_2_second - dmg_2_first) / MAX_HIT_POINTS
        r[second] += (dmg_2_first - dmg_2_second) / MAX_HIT_POINTS

        active_not_fainted = not (first_pkm.fainted() or second_pkm.fainted())

        # battle
        first_can_attack = active_not_fainted and not first_pkm.paralyzed() and not first_pkm.asleep() and not \
            first_confusion_damage
        if self.debug and not first_can_attack:
            self.log += 'CANNOT MOVE: Trainer %d cannot move\n' % first
        dmg_2_second, hp_2_first = self.__perform_pkm_attack(first, actions[first]) if first_can_attack else (0., 0.)

        active_not_fainted = not (first_pkm.fainted() or second_pkm.fainted())

        second_can_attack = active_not_fainted and not second_pkm.paralyzed() and not second_pkm.asleep() and not \
            second_confusion_damage
        if self.debug and not second_can_attack:
            self.log += 'CANNOT MOVE: Trainer %d cannot move\n' % second
        dmg_2_first, hp_2_second = self.__perform_pkm_attack(second, actions[second]) if second_can_attack else (0., 0.)

        r[first] += (dmg_2_second + hp_2_first - dmg_2_first) / MAX_HIT_POINTS + float(second_pkm.fainted())
        r[second] += (dmg_2_first + hp_2_second - dmg_2_second) / MAX_HIT_POINTS + float(first_pkm.fainted())

        # get post battle effects damage
        dmg_2_first = self.__get_post_battle_damage(first) if first_can_attack else 0.
        dmg_2_second = self.__get_post_battle_damage(second) if second_can_attack else 0.

        r[first] += (dmg_2_second - dmg_2_first) / MAX_HIT_POINTS
        r[second] += (dmg_2_first - dmg_2_second) / MAX_HIT_POINTS

        # process all post battle effects
        self.__process_post_battle_effects()

        # switch fainted pkm
        dmg_2_first, dmg_2_second = self.__switch_fainted_pkm()

        r[first] += (dmg_2_second - dmg_2_first) / MAX_HIT_POINTS
        r[second] += (dmg_2_first - dmg_2_second) / MAX_HIT_POINTS

        # check if battle ended
        t[first] = first_team.fainted()
        t[second] = second_team.fainted()

        r[first] += float(t[first])
        r[second] += float(t[second])

        finished = t[0] or t[1]

        if finished:
            self.winner = 1 if t[0] else 0

            if self.debug:
                self.log += '\nTRAINER %s %s\n' % (0, 'Lost' if self.teams[0].fainted() else 'Won')
                self.log += 'TRAINER %s %s\n' % (1, 'Lost' if self.teams[1].fainted() else 'Won')

        return [self.trainer_view[0].encode(), self.trainer_view[1].encode()], r, finished, self.trainer_view

    def reset(self):
        self.weather = WeatherCondition.CLEAR
        self.n_turns_no_clear = 0
        self.turn = 0
        self.winner = -1
        self.switched = [False, False]

        if self.team_generator is not None:
            self.teams[0] = self.team_generator.get_team(0)
            self.teams[1] = self.team_generator.get_team(1)

        for team in self.teams:
            team.reset()

        for trainer_view in self.trainer_view:
            trainer_view.reset()

        if self.debug:
            self.log = 'TRAINER 0\n' + str(self.teams[0])
            self.log += '\nTRAINER 1\n' + str(self.teams[1])

        return [self.trainer_view[0].encode(), self.trainer_view[1].encode()]

    def render(self, mode='human'):
        print(self.log)

    def __process_switch_pkms(self, actions: List[int]):
        """
        Switch pkm if players chosen to do so.

        :param actions: players actions
        :return:
        """
        for i, team in enumerate(self.teams):
            pos = actions[i] - N_MOVES
            if 0 <= pos < (team.size() - 1):
                if not team.party[pos].fainted():
                    new_active, old_active = team.switch(pos)
                    self.switched[i] = True
                    if self.debug:
                        self.log += 'SWITCH: Trainer %s switches %s with %s in party\n' % (i, old_active, new_active)
                elif self.debug:
                    self.log += 'SWITCH FAILED: Trainer %d fails to switch\n' % i
            elif self.debug and pos >= (team.size() - 1):
                self.log += 'INVALID SWITCH: Trainer %d fails to switch\n' % i

    def __get_entry_hazard_damage(self, t_id: int) -> float:
        """
        Get triggered damage to be dealt to a switched pkm.

        :param: t_id: owner trainer
        :return: damage to first pkm, damage to second pkm
        """
        damage = 0.
        team = self.teams[t_id]
        pkm = team.active
        spikes = team.entry_hazard[PkmEntryHazard.SPIKES]

        # Spikes damage
        if spikes and pkm.type != PkmType.FLYING and self.switched[t_id]:
            before_hp = pkm.hp
            pkm.hp -= STATE_DAMAGE if spikes <= 1 else SPIKES_2 if spikes == 2 else SPIKES_3
            pkm.hp = 0. if pkm.hp < 0. else pkm.hp
            damage = before_hp - pkm.hp
            self.switched[t_id] = False
            if self.debug and damage > 0.:
                self.log += 'ENTRY HAZARD DAMAGE: %s takes %s entry hazard damage from spikes, hp reduces from %s to ' \
                            '%s\n' % (str(pkm), damage, before_hp, pkm.hp)

        return damage

    def __process_pre_battle_effects(self):
        """
        Process all pre battle effects.

        """
        # for all trainers
        for i in range(len(self.teams)):

            team = self.teams[i]
            pkm = team.active

            # check if active pkm should be no more confused
            if team.confused:
                team.n_turns_confused += 1
                if random.uniform(0, 1) <= 0.5 or team.n_turns_confused == 4:
                    team.confused = False
                    team.n_turns_confused = 0
                    if self.debug:
                        self.log += 'STATUS: Trainer %s\'s %s is no longer confused\n' % (i, str(pkm))

            # check if active pkm should be no more asleep
            if pkm.asleep():
                pkm.n_turns_asleep += 1
                if random.uniform(0, 1) <= 0.5 or pkm.n_turns_asleep == 4:
                    pkm.status = PkmStatus.NONE
                    pkm.n_turns_asleep = 0
                    if self.debug:
                        self.log += 'STATUS: Trainer %s\'s %s is no longer asleep\n' % (i, str(pkm))

    def __process_post_battle_effects(self):
        """
        Process all post battle effects.

        """
        if self.weather != WeatherCondition.CLEAR:
            self.n_turns_no_clear += 1

            # clear weather if appropriated
            if self.n_turns_no_clear > 5:
                self.weather = WeatherCondition.CLEAR
                self.n_turns_no_clear = 0
                if self.debug:
                    self.log += 'STATE: The weather is clear\n'

    def __get_post_battle_damage(self, t_id: int) -> float:
        """
        Get triggered damage to be dealt to switched pkm.

        :param: t_id: owner trainer
        :return: damage to pkm
        """
        pkm = self.teams[t_id].active
        state_damage = 0.

        if self.weather == WeatherCondition.SANDSTORM and (
                pkm.type != PkmType.ROCK and pkm.type != PkmType.GROUND and pkm.type != PkmType.STEEL):
            state_damage = STATE_DAMAGE
        elif self.weather == WeatherCondition.HAIL and (pkm.type != PkmType.ICE):
            state_damage = STATE_DAMAGE

        before_hp = pkm.hp
        pkm.hp -= state_damage
        pkm.hp = 0. if pkm.hp < 0. else pkm.hp
        damage = before_hp - pkm.hp

        if self.debug and state_damage > 0.:
            self.log += 'STATE DAMAGE: %s takes %s weather damage from sandstorm/hail hp reduces from %s to %s\n' % (
                str(pkm), damage, before_hp, pkm.hp)

        if pkm.status == PkmStatus.POISONED or pkm.status == PkmStatus.BURNED:
            state_damage = STATE_DAMAGE

            before_hp = pkm.hp
            pkm.hp -= state_damage
            pkm.hp = 0. if pkm.hp < 0. else pkm.hp
            damage = before_hp - pkm.hp

            if self.debug and damage > 0.:  # TODO burned and frozen
                self.log += 'STATE DAMAGE: %s takes %s state damage from poison, hp reduces from %s to %s\n' % (
                    str(pkm), damage, before_hp, pkm.hp)

        return damage

    def __get_attack_order(self, actions) -> Tuple[int, int]:
        """
        Get attack order for this turn.
        Priority is given to the pkm with highest speed_stage. Otherwise random.

        :return: tuple with first and second trainer to perform attack
        """
        action0 = actions[0]
        action1 = actions[1]
        speed0 = self.teams[0].stage[PkmStat.SPEED] + (
            self.teams[0].active.moves[action0].priority if action0 < N_MOVES else 0)
        speed1 = self.teams[1].stage[PkmStat.SPEED] + (
            self.teams[1].active.moves[action1].priority if action1 < N_MOVES else 0)
        if speed0 > speed1:
            order = [0, 1]
        elif speed1 < speed0:
            order = [1, 0]
        else:
            # random attack order
            order = [0, 1]
            np.random.shuffle(order)

        return order[0], order[1]

    class TrainerView:

        def __init__(self, engine, t_id: int = 0):
            self.engine = engine
            self.t_id = t_id
            self.team: PkmTeam = engine.teams[self.t_id]
            self.opp: PkmTeam = engine.teams[not self.t_id]

        def get_active(self) -> Tuple[PkmType, float, PkmStatus, bool]:
            return self.team.active.type, self.team.active.hp, self.team.active.status, self.team.confused

        def get_opponent(self) -> Tuple[PkmType, float, PkmStatus, bool]:
            return self.opp.active.type, self.opp.active.hp, self.opp.active.status, self.opp.confused

        def get_party(self, pos: int = 0) -> Tuple[PkmType, float, PkmStatus]:
            pkm = self.team.party[pos]
            return pkm.type, pkm.hp, pkm.status

        def get_n_party(self) -> int:
            return len(self.team.party)

        def get_active_move(self, pos: int = 0) -> Tuple[float, PkmType, str]:
            move = self.team.active.moves[pos]
            return move.power, move.type, move.name

        def get_n_moves(self) -> int:
            return len(self.team.active.moves)

        def get_stage(self, stat: PkmStat = PkmStat.ATTACK, t_id: int = 0) -> int:
            if t_id == 0:
                return self.team.stage[stat]
            else:
                return self.opp.stage[stat]

        def get_entry_hazard(self, hazard: PkmEntryHazard = PkmEntryHazard.SPIKES) -> int:
            return self.team.stage[hazard]

        def get_weather(self) -> WeatherCondition:
            return self.engine.weather

        def reset(self):
            self.team: PkmTeam = self.engine.teams[self.t_id]
            self.opp: PkmTeam = self.engine.teams[not self.t_id]

        def encode(self):
            """
            Encode Game state.

            :return: encoded game state
            """
            e = []
            # active pkms
            for team in self.engine.teams:
                e += one_hot(team.active.type, N_TYPES)
                e += [team.active.hp / MAX_HIT_POINTS]
                e += one_hot(team.active.status, N_STATUS)
                e += one_hot(team.confused, 2)
                # stages
                for stat in range(N_STATS):
                    e += one_hot(team.stage[stat], N_STAGES)
                # entry hazards
                for hazard in range(N_ENTRY_HAZARD):
                    e += one_hot(team.entry_hazard[hazard], N_HAZARD_STAGES)
            # party pkm
            for pos in range(len(self.team.party)):
                party = self.team.party[pos]
                e += one_hot(party.type, N_TYPES)
                e += [party.hp / MAX_HIT_POINTS]
                e += one_hot(party.status, N_STATUS)
            # active moves
            for pos in range(len(self.team.active.moves)):
                move = self.team.active.moves[pos]
                e += [move.power / MAX_HIT_POINTS]
                e += [move.acc]
                e += [float(move.pp / MOVE_MAX_PP)]
                e += [move.priority]
                e += one_hot(move.type, N_TYPES)
            # weather
            e += one_hot(self.engine.weather, N_WEATHER)
            return e

    def __create_trainer_view(self, t_id: int = 0) -> TrainerView:
        return PkmBattleEnv.TrainerView(self, t_id)

    class PkmMoveView:

        def __init__(self, engine):
            self.__engine = engine
            self._damage: float = 0.
            self._recover: float = 0.
            self._team: List[PkmTeam] = []
            self._active: List[Pkm] = []

        def set_weather(self, weather: WeatherCondition):
            if weather != self.__engine.weather:
                self.__engine.weather = weather
                self.__engine.n_turns_no_clear = 0
                if self.__engine.debug:
                    self.__engine.log += 'STATE: The weather is now %s\n' % weather.name

        def set_fixed_damage(self, damage: float):
            self._damage = damage

        def set_recover(self, recover: float):
            self._recover = recover

        def set_status(self, status: PkmStatus, t_id: int = 1):
            pkm = self._active[t_id]
            team = self._team[t_id]
            if status == PkmStatus.PARALYZED and pkm.type != PkmType.ELECTRIC and pkm.type != PkmType.GROUND and \
                    pkm.status != PkmStatus.PARALYZED:
                pkm.status = PkmStatus.PARALYZED
                if self.__engine.debug:
                    self.__engine.log += 'STATUS: %s was paralyzed\n' % (str(pkm))
            elif status == PkmStatus.POISONED and pkm.type != PkmType.POISON and pkm.type != PkmType.STEEL and \
                    pkm.status != PkmStatus.POISONED:
                pkm.status = PkmStatus.POISONED
                if self.__engine.debug:
                    self.__engine.log += 'STATUS: %s was poisoned\n' % (str(pkm))
            elif status == PkmStatus.SLEEP and pkm.status != PkmStatus.SLEEP:
                pkm.status = PkmStatus.SLEEP
                pkm.n_turns_asleep = 0
                if self.__engine.debug:
                    self.__engine.log += 'STATUS: %s is now asleep\n' % (str(pkm))
            elif not team.confused:
                team.confused = True
                if self.__engine.debug:
                    self.__engine.log += 'STATUS: %s is now confused\n' % (str(pkm))

        def set_stage(self, stat: PkmStat = PkmStat.ATTACK, delta_stage: int = 1, t_id: int = 1):
            assert delta_stage != 0
            team = self._team[t_id]
            if MIN_STAGE < team.stage[stat] < MAX_STAGE:
                team.stage[stat] += delta_stage
                if self.__engine.debug:
                    self.__engine.log += 'STAGE: %s %s %s\n' % (
                        str(team.active), stat.name, 'increased' if delta_stage > 0 else 'decreased')

        def set_entry_hazard(self, hazard: PkmEntryHazard = PkmEntryHazard.SPIKES, t_id: int = 1):
            team = self.__engine.teams[t_id]
            team.entry_hazard[hazard] += 1
            if team.entry_hazard[hazard] >= N_HAZARD_STAGES:
                team.entry_hazard[hazard] = N_HAZARD_STAGES - 1
            elif self.__engine.debug:
                self.__engine.log += 'ENTRY HAZARD: Trainer %s gets spikes\n' % (str(t_id))

        @property
        def recover(self):
            return self._recover

        @property
        def damage(self):
            return self._damage

    def __get_fixed_damage(self) -> float:
        damage = self.move_view.damage
        self.move_view._damage = 0.
        return damage

    def __get_recover(self) -> float:
        recover = self.move_view.recover
        self.move_view._recover = 0.
        return recover

    def __create_pkm_move_view(self):
        return PkmBattleEnv.PkmMoveView(self)

    def __get_attack_dmg_rcvr(self, t_id: int, m_id: int) -> Tuple[float, float]:
        """
        Get damage and recover done by an attack m_id of active pkm of trainer t_id

        :param t_id: trainer of the active pkm
        :param m_id: move of the active pkm
        :return: damage, recover
        """

        team = self.teams[t_id]
        pkm = team.active
        move = pkm.moves[m_id]

        if move.pp > 0:
            move.pp -= 1
        else:
            move = Struggle

        if move.acc <= random.random():
            if self.debug:
                self.log += 'MOVE FAILS: Trainer %s with %s fails %s\n' % (t_id, str(pkm), str(move))
            return 0., 0.

        opp = not t_id
        opp_team = self.teams[opp]
        opp_pkm = opp_team.active

        if self.debug:
            self.log += 'MOVE: Trainer %s with %s uses %s\n' % (t_id, str(pkm), str(move))

        self.move_view._team = [team, opp_team]
        self.move_view._active = [pkm, opp_pkm]
        move.effect(self.move_view)

        # set recover
        recover = self.__get_recover()

        # calculate damage
        fixed_damage = self.__get_fixed_damage()
        if fixed_damage > 0. and TYPE_CHART_MULTIPLIER[move.type][opp_pkm.type] > 0.:
            damage = fixed_damage
        else:
            stab = 1.5 if move.type == pkm.type else 1.
            if (move.type == PkmType.WATER and self.weather == WeatherCondition.RAIN) or (
                    move.type == PkmType.FIRE and self.weather == WeatherCondition.SUNNY):
                weather = 1.5
            elif (move.type == PkmType.WATER and self.weather == WeatherCondition.SUNNY) or (
                    move.type == PkmType.FIRE and self.weather == WeatherCondition.RAIN):
                weather = .5
            else:
                weather = 1.
            stage_level = team.stage[PkmStat.ATTACK] - opp_team.stage[PkmStat.DEFENSE]
            stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
            damage = TYPE_CHART_MULTIPLIER[move.type][opp_pkm.type] * stab * weather * stage * move.power

        return round(damage), round(recover)

    def __perform_pkm_attack(self, t_id: int, m_id: int) -> Tuple[float, float]:
        """
        Perform a pkm attack

        :param t_id: trainer
        :param m_id: move
        :return: reward, recover
        """
        damage, recover = 0., 0.

        if m_id < N_MOVES:
            opponent = not t_id
            pkm = self.teams[t_id].active
            before_hp = pkm.hp
            opp_pkm = self.teams[opponent].active
            before_opp_hp = opp_pkm.hp

            # get damage and recover values from attack
            damage_2_deal, health_2_recover = self.__get_attack_dmg_rcvr(t_id, m_id)

            # perform recover
            pkm.hp += health_2_recover
            pkm.hp = MAX_HIT_POINTS if pkm.hp > MAX_HIT_POINTS else pkm.hp
            recover = pkm.hp - before_hp
            if self.debug and recover > 0.:
                self.log += 'RECOVER: recovers %s\n' % recover
            elif self.debug and recover < 0.:
                self.log += 'RECOIL DAMAGE: gets %s recoil damage\n' % -recover

            # perform damage
            opp_pkm.hp -= damage_2_deal
            opp_pkm.hp = 0. if opp_pkm.hp < 0. else opp_pkm.hp
            damage = before_opp_hp - opp_pkm.hp
            if self.debug and damage > 0.:
                self.log += 'DAMAGE: deals %s damage, hp reduces from %s to %s for %s\n' % (
                    damage, before_opp_hp, opp_pkm.hp, str(opp_pkm))

        return damage, recover

    def __get_pre_combat_damage(self, t_id: int) -> float:
        """
        Check if trainer t_id active pkm is confused this turn and cannot move and take damage.

        :param t_id: trainer
        :return: 0. if not confused or damage to take if confused
        """
        return STATE_DAMAGE if self.teams[t_id].confused and random.uniform(0, 1) <= 0.33 else 0.

    def __switch_fainted_pkm(self) -> Tuple[float, float]:
        """
        Recursive damage dealt to fainted switched pkm, while faiting for entry hazard.

        :return: damage to pkm 0, damage to pkm 1
        """
        damage0, damage1 = 0., 0.
        self.switched = [False, False]
        team0 = self.teams[0]
        team1 = self.teams[1]
        pkm0 = self.teams[0].active
        pkm1 = self.teams[1].active
        if pkm0.fainted():
            if self.debug:
                self.log += 'FAINTED: %s\n' % (str(pkm0))
            team0.switch(-1)
        if pkm1.fainted():
            if self.debug:
                self.log += 'FAINTED: %s\n' % (str(pkm1))
            team1.switch(-1)
        if not pkm0.fainted():
            damage0 = self.__get_entry_hazard_damage(0)
        if not pkm1.fainted():
            damage1 = self.__get_entry_hazard_damage(1)
        d0, d1 = 0., 0.
        if (pkm0.fainted() or pkm1.fainted()) and (not team0.fainted() and not team1.fainted()):
            d0, d1 = self.__switch_fainted_pkm()
        return damage0 + d0, damage1 + d1


class BattleEngine:

    def __init__(self, bp0: BattlePolicy, bp1: BattlePolicy, team0: PkmTeam, team1: PkmTeam, rec: Recorder, debug=True,
                 render=True, n_battles=N_BATTLES):
        self.env = PkmBattleEnv(teams=[team0, team1], debug=debug)
        self.bp0 = bp0
        self.bp1 = bp1
        self.rec = rec
        self.ep = 0
        self.step = 0
        self.n_battles = n_battles
        self.render = render
        self.s = None
        self.v = None
        self.t = True

    def run_step(self):  # TODO return type
        if self.t:
            self.s = self.env.reset()
            self.v = self.env.trainer_view
            self.ep += 1
            if self.render:
                self.env.render()
        o0 = self.s[0] if self.bp0.requires_encode() else self.v[0]
        o1 = self.s[1] if self.bp1.requires_encode() else self.v[1]
        a = [self.bp0.get_action(o0), self.bp1.get_action(o1)]
        self.rec.record((self.s[0], a[0], self.ep))  # TODO record both players
        self.s, r, self.t, self.v = self.env.step(a)
        if self.render:
            self.env.render()
        return r

    def match_completed(self) -> bool:
        return self.t and self.ep == self.n_battles

    def terminate(self):
        self.rec.save()
        self.bp0.close()
        self.bp1.close()